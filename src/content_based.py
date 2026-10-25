"""
CineMatch: Content-Based Recommendation Engine
Implements TF-IDF vectorization and cosine similarity over enriched metadata
(genres, plot overviews, cast, director, user tags, and keywords) with explainability.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedRecommender:
    """
    Computes movie similarities and user-profile recommendations using TF-IDF and Cosine Similarity.
    """
    def __init__(self, max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range
        )
        self.df_movies: Optional[pd.DataFrame] = None
        self.tfidf_matrix = None
        self.movie_id_to_idx: Dict[int, int] = {}
        self.idx_to_movie_id: Dict[int, int] = {}

    def fit(self, df_movies: pd.DataFrame):
        """
        Fits TF-IDF vectorizer on the metadata soup of all catalog movies.
        """
        self.df_movies = df_movies.reset_index(drop=True)
        self.movie_id_to_idx = {row.movieId: idx for idx, row in self.df_movies.iterrows()}
        self.idx_to_movie_id = {idx: row.movieId for idx, row in self.df_movies.iterrows()}
        
        # Fit vectorizer on metadata soup
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df_movies['soup'])
        return self

    def get_similar_movies(self, movie_id: int, top_n: int = 10, min_similarity: float = 0.02) -> List[Dict[str, Any]]:
        """
        Retrieves top N movies most similar to the given seed movie.
        """
        if movie_id not in self.movie_id_to_idx:
            return []

        seed_idx = self.movie_id_to_idx[movie_id]
        seed_vector = self.tfidf_matrix[seed_idx:seed_idx + 1]

        # Compute cosine similarity (linear kernel over normalized TF-IDF)
        sim_scores = linear_kernel(seed_vector, self.tfidf_matrix).flatten()

        # Sort indices in descending order, excluding the seed movie itself
        ranked_indices = sim_scores.argsort()[::-1]
        
        results = []
        seed_movie = self.df_movies.iloc[seed_idx]
        seed_genres = set(seed_movie['genres_list'])

        for idx in ranked_indices:
            if idx == seed_idx:
                continue
            
            score = float(sim_scores[idx])
            if score < min_similarity:
                break
                
            rec_movie = self.df_movies.iloc[idx]
            rec_genres = set(rec_movie['genres_list'])
            shared_genres = list(seed_genres.intersection(rec_genres))

            # Build explainability snippet
            reasons = []
            if shared_genres:
                reasons.append(f"Shares {', '.join(shared_genres[:3])}")
            if seed_movie.get('director') and seed_movie['director'] == rec_movie.get('director'):
                reasons.append(f"Directed by {seed_movie['director']}")
            if not reasons:
                reasons.append("Thematic storyline and keyword overlap")

            results.append({
                "movieId": int(rec_movie['movieId']),
                "title": rec_movie['clean_title'],
                "year": rec_movie['year'],
                "genres": rec_movie['genres'],
                "genres_list": rec_movie['genres_list'],
                "primary_genre": rec_movie['primary_genre'],
                "rating_mean": float(rec_movie['rating_mean']),
                "rating_count": int(rec_movie['rating_count']),
                "tmdbId": rec_movie.get('tmdbId'),
                "similarity_score": round(score, 4),
                "similarity_pct": f"{round(score * 100, 1)}%",
                "explanation": " • ".join(reasons)
            })

            if len(results) >= top_n:
                break

        return results

    def get_recommendations_for_user_profile(
        self,
        user_ratings: Dict[int, float],
        top_n: int = 10,
        exclude_rated: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Builds a synthesized user preference profile from rated movies
        and finds catalog movies that best match the user's aggregated taste vector.
        """
        if not user_ratings or self.tfidf_matrix is None:
            return []

        # Construct weighted user profile vector:
        # Weight ratings above 2.5 positively, below 2.5 negatively
        user_vector = np.zeros((1, self.tfidf_matrix.shape[1]))
        total_weight = 0.0

        for mid, rating in user_ratings.items():
            if mid in self.movie_id_to_idx:
                idx = self.movie_id_to_idx[mid]
                # Center around neutral rating (3.0): e.g. 5.0 -> +2.0, 4.0 -> +1.0, 2.0 -> -1.0
                weight = max(0.2, rating - 2.5)
                movie_tfidf = self.tfidf_matrix[idx].toarray()
                user_vector += weight * movie_tfidf
                total_weight += weight

        if total_weight > 0:
            user_vector /= total_weight
            # L2 normalize
            norm = np.linalg.norm(user_vector)
            if norm > 0:
                user_vector /= norm

        # Compute cosine similarity against all movies
        sim_scores = linear_kernel(user_vector, self.tfidf_matrix).flatten()
        ranked_indices = sim_scores.argsort()[::-1]

        rated_mids = set(user_ratings.keys()) if exclude_rated else set()
        results = []

        for idx in ranked_indices:
            mid = int(self.idx_to_movie_id[idx])
            if mid in rated_mids:
                continue

            score = float(sim_scores[idx])
            rec_movie = self.df_movies.iloc[idx]

            results.append({
                "movieId": mid,
                "title": rec_movie['clean_title'],
                "year": rec_movie['year'],
                "genres": rec_movie['genres'],
                "genres_list": rec_movie['genres_list'],
                "primary_genre": rec_movie['primary_genre'],
                "rating_mean": float(rec_movie['rating_mean']),
                "rating_count": int(rec_movie['rating_count']),
                "tmdbId": rec_movie.get('tmdbId'),
                "similarity_score": round(score, 4),
                "similarity_pct": f"{round(score * 100, 1)}%",
                "explanation": f"Matches your taste in {rec_movie['primary_genre']} & favorite story themes"
            })

            if len(results) >= top_n:
                break

        return results

    def get_similarity_score(self, movie_id_1: int, movie_id_2: int) -> float:
        """Computes pairwise cosine similarity between two movies."""
        if movie_id_1 not in self.movie_id_to_idx or movie_id_2 not in self.movie_id_to_idx:
            return 0.0
        idx1 = self.movie_id_to_idx[movie_id_1]
        idx2 = self.movie_id_to_idx[movie_id_2]
        v1 = self.tfidf_matrix[idx1:idx1+1]
        v2 = self.tfidf_matrix[idx2:idx2+1]
        return float(linear_kernel(v1, v2)[0, 0])
