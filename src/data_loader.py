"""
CineMatch: Data Loader and Preprocessor
Handles automated acquisition, cleaning, feature engineering, and EDA aggregation
for the MovieLens latest-small dataset enriched with TMDB identifiers and metadata.
"""

import os
import re
import io
import zipfile
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from .curated_data import CURATED_TMDB_DATA

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def clean_movie_title(raw_title: str) -> Tuple[str, str]:
    """
    Extracts the 4-digit release year and cleans grammatical suffixes
    like 'Shawshank Redemption, The' into 'The Shawshank Redemption'.
    """
    if not isinstance(raw_title, str):
        return str(raw_title), ""
    
    # Extract 4-digit year inside parentheses
    year_match = re.search(r'\((\d{4})\)', raw_title)
    year = year_match.group(1) if year_match else ""
    
    # Strip year and any following tokens from the title
    clean = re.sub(r'\s*\(\d{4}\).*$', '', raw_title).strip()
    
    # Invert trailing articles for standard reading order
    for article in ['The', 'A', 'An', 'El', 'La', 'Le', 'Les', 'Der', 'Die', 'Das']:
        if clean.endswith(f', {article}'):
            clean = f'{article} ' + clean[:-len(f', {article}')]
            break
            
    return clean, year

class DataLoader:
    """
    Loads, cleans, enriches, and caches MovieLens dataset files.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.ml_dir = os.path.join(data_dir, "ml-latest-small")
        self.processed_cache = os.path.join(data_dir, "processed_movies.csv")
        
        self.df_movies: Optional[pd.DataFrame] = None
        self.df_ratings: Optional[pd.DataFrame] = None
        self.df_links: Optional[pd.DataFrame] = None
        self.df_tags: Optional[pd.DataFrame] = None
        
        self._ensure_dataset_available()
        self.load_and_preprocess()

    def _ensure_dataset_available(self):
        """Downloads and unzips MovieLens dataset if missing."""
        os.makedirs(self.data_dir, exist_ok=True)
        movies_csv = os.path.join(self.ml_dir, "movies.csv")
        ratings_csv = os.path.join(self.ml_dir, "ratings.csv")
        
        if not (os.path.exists(movies_csv) and os.path.exists(ratings_csv)):
            try:
                resp = requests.get(MOVIELENS_URL, verify=False, timeout=60)
                if resp.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                        z.extractall(self.data_dir)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to automatically download MovieLens dataset: {e}. "
                    f"Please place 'ml-latest-small' inside '{self.data_dir}/'."
                )

    def load_and_preprocess(self):
        """Loads raw CSVs and builds the unified enriched movie catalog."""
        # 1. Load raw files
        movies_path = os.path.join(self.ml_dir, "movies.csv")
        ratings_path = os.path.join(self.ml_dir, "ratings.csv")
        links_path = os.path.join(self.ml_dir, "links.csv")
        tags_path = os.path.join(self.ml_dir, "tags.csv")
        
        self.df_movies = pd.read_csv(movies_path)
        self.df_ratings = pd.read_csv(ratings_path)
        self.df_links = pd.read_csv(links_path) if os.path.exists(links_path) else pd.DataFrame(columns=['movieId', 'imdbId', 'tmdbId'])
        self.df_tags = pd.read_csv(tags_path) if os.path.exists(tags_path) else pd.DataFrame(columns=['userId', 'movieId', 'tag'])

        # 2. Extract year and clean titles
        cleaned_titles_and_years = [clean_movie_title(t) for t in self.df_movies['title']]
        self.df_movies['clean_title'] = [c[0] for c in cleaned_titles_and_years]
        self.df_movies['year'] = [c[1] for c in cleaned_titles_and_years]

        # 3. Clean and parse genres
        self.df_movies['genres_list'] = self.df_movies['genres'].apply(
            lambda g: [] if g == "(no genres listed)" or pd.isna(g) else g.split('|')
        )
        self.df_movies['primary_genre'] = self.df_movies['genres_list'].apply(
            lambda gl: gl[0] if len(gl) > 0 else 'General'
        )

        # 4. Aggregate user tags per movie
        if not self.df_tags.empty:
            tag_agg = self.df_tags.groupby('movieId')['tag'].apply(
                lambda tags: " ".join([str(t).lower() for t in tags if pd.notna(t)])
            ).reset_index().rename(columns={'tag': 'user_tags'})
            self.df_movies = pd.merge(self.df_movies, tag_agg, on='movieId', how='left')
        else:
            self.df_movies['user_tags'] = ""
        self.df_movies['user_tags'] = self.df_movies['user_tags'].fillna("")

        # 5. Merge TMDB links
        if not self.df_links.empty:
            self.df_movies = pd.merge(self.df_movies, self.df_links[['movieId', 'tmdbId', 'imdbId']], on='movieId', how='left')
        else:
            self.df_movies['tmdbId'] = np.nan
            self.df_movies['imdbId'] = np.nan

        # 6. Aggregate rating counts and averages
        rating_stats = self.df_ratings.groupby('movieId').agg(
            rating_count=('rating', 'count'),
            rating_mean=('rating', 'mean')
        ).reset_index()
        self.df_movies = pd.merge(self.df_movies, rating_stats, on='movieId', how='left')
        self.df_movies['rating_count'] = self.df_movies['rating_count'].fillna(0).astype(int)
        self.df_movies['rating_mean'] = self.df_movies['rating_mean'].fillna(0.0).round(2)

        # Compute Bayesian Weighted Rating: WR = (v / (v + m)) * R + (m / (v + m)) * C
        C = self.df_ratings['rating'].mean()
        m = 25  # Minimum votes threshold
        v = self.df_movies['rating_count']
        R = self.df_movies['rating_mean']
        self.df_movies['bayesian_rating'] = ((v / (v + m)) * R + (m / (v + m)) * C).round(3)

        # 7. Merge Curated TMDB Metadata & synthesize metadata soup
        overviews = []
        directors = []
        casts = []
        keywords_list = []
        soups = []

        for _, row in self.df_movies.iterrows():
            tmdb_id = row['tmdbId']
            clean_id = int(tmdb_id) if pd.notna(tmdb_id) else None
            
            if clean_id and clean_id in CURATED_TMDB_DATA:
                c = CURATED_TMDB_DATA[clean_id]
                ov = c.get('overview', '')
                d = c.get('director', '')
                ca = " ".join(c.get('cast', []))
                kw = " ".join(c.get('keywords', []))
            else:
                ov = f"{row['clean_title']} is a notable {row['primary_genre']} film released in {row['year']}."
                d = ""
                ca = ""
                kw = ""

            overviews.append(ov)
            directors.append(d)
            casts.append(ca)
            keywords_list.append(kw)

            # Metadata soup for TF-IDF: combines title, genres, director, cast, keywords, user tags, overview
            soup_tokens = [
                row['clean_title'],
                " ".join(row['genres_list']),
                d,
                ca,
                kw,
                row['user_tags'],
                ov
            ]
            soups.append(" ".join([t for t in soup_tokens if t]).lower())

        self.df_movies['overview'] = overviews
        self.df_movies['director'] = directors
        self.df_movies['cast'] = casts
        self.df_movies['keywords'] = keywords_list
        self.df_movies['soup'] = soups

    # ================= EDA & Statistics Helpers ================= #
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Returns fundamental dataset parameters and matrix sparsity."""
        n_users = self.df_ratings['userId'].nunique()
        n_movies = self.df_movies['movieId'].nunique()
        n_ratings = len(self.df_ratings)
        sparsity = (1.0 - (n_ratings / (n_users * n_movies))) * 100

        return {
            "total_movies": n_movies,
            "total_users": n_users,
            "total_ratings": n_ratings,
            "mean_rating": round(float(self.df_ratings['rating'].mean()), 2),
            "median_rating": float(self.df_ratings['rating'].median()),
            "std_rating": round(float(self.df_ratings['rating'].std()), 2),
            "sparsity_pct": round(sparsity, 2),
            "min_user_ratings": int(self.df_ratings['userId'].value_counts().min()),
            "max_user_ratings": int(self.df_ratings['userId'].value_counts().max()),
            "median_user_ratings": int(self.df_ratings['userId'].value_counts().median())
        }

    def get_genre_distribution(self) -> pd.DataFrame:
        """Calculates frequency and percentage for every genre in catalog."""
        genre_counts = {}
        for glist in self.df_movies['genres_list']:
            for g in glist:
                genre_counts[g] = genre_counts.get(g, 0) + 1
        
        df_g = pd.DataFrame(list(genre_counts.items()), columns=['genre', 'count'])
        df_g = df_g.sort_values(by='count', ascending=False).reset_index(drop=True)
        df_g['percentage'] = (df_g['count'] / len(self.df_movies) * 100).round(1)
        return df_g

    def get_rating_distribution(self) -> pd.DataFrame:
        """Returns distribution of rating values (0.5 to 5.0)."""
        counts = self.df_ratings['rating'].value_counts().sort_index().reset_index()
        counts.columns = ['rating', 'count']
        counts['percentage'] = (counts['count'] / len(self.df_ratings) * 100).round(2)
        return counts

    def get_popularity_distribution(self) -> pd.DataFrame:
        """Returns sorted movie rating frequencies for long-tail curve analysis."""
        return self.df_movies[['movieId', 'clean_title', 'rating_count', 'rating_mean']]\
            .sort_values(by='rating_count', ascending=False)\
            .reset_index(drop=True)

    def get_top_rated_movies(self, min_ratings: int = 50, top_n: int = 20) -> pd.DataFrame:
        """Returns top movies ranked by Bayesian weighted rating score."""
        filtered = self.df_movies[self.df_movies['rating_count'] >= min_ratings]
        return filtered.sort_values(by='bayesian_rating', ascending=False).head(top_n)

    def get_trending_movies(self, top_n: int = 20) -> pd.DataFrame:
        """Returns most frequently rated movies in catalog."""
        return self.df_movies.sort_values(by='rating_count', ascending=False).head(top_n)
