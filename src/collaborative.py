"""
CineMatch: Collaborative Filtering Engine
Implements Matrix Factorization via SVD (Singular Value Decomposition) and Memory-Based KNN,
with a transparent pure-NumPy Funk-SVD implementation for algorithmic inspection.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from surprise import SVD, KNNWithMeans, Reader, Dataset

class FunkSVDRecommender:
    """
    Pure NumPy implementation of Simon Funk's Regularized SVD with Stochastic Gradient Descent (SGD).
    Objective:
      min sum (r_ui - (mu + b_u + b_i + p_u^T q_i))^2 + lambda * (b_u^2 + b_i^2 + ||p_u||^2 + ||q_i||^2)
    """
    def __init__(self, n_factors: int = 20, n_epochs: int = 15, lr: float = 0.005, reg: float = 0.02, random_state: int = 42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state
        
        self.mu: float = 0.0
        self.b_u: np.ndarray = np.array([])
        self.b_i: np.ndarray = np.array([])
        self.P: np.ndarray = np.array([])  # User latent factors
        self.Q: np.ndarray = np.array([])  # Item latent factors
        
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}

    def fit(self, df_ratings: pd.DataFrame):
        rng = np.random.RandomState(self.random_state)
        self.mu = float(df_ratings['rating'].mean())
        
        unique_users = df_ratings['userId'].unique()
        unique_items = df_ratings['movieId'].unique()
        
        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.item_to_idx = {m: i for i, m in enumerate(unique_items)}
        self.idx_to_item = {i: m for m, i in self.item_to_idx.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))
        
        u_indices = df_ratings['userId'].map(self.user_to_idx).values
        i_indices = df_ratings['movieId'].map(self.item_to_idx).values
        ratings = df_ratings['rating'].values
        
        # SGD Optimization
        for epoch in range(self.n_epochs):
            for u_idx, i_idx, r in zip(u_indices, i_indices, ratings):
                dot_val = np.dot(self.P[u_idx], self.Q[i_idx])
                pred = self.mu + self.b_u[u_idx] + self.b_i[i_idx] + dot_val
                err = r - pred
                
                # Bias updates
                self.b_u[u_idx] += self.lr * (err - self.reg * self.b_u[u_idx])
                self.b_i[i_idx] += self.lr * (err - self.reg * self.b_i[i_idx])
                
                # Factor updates
                p_old = self.P[u_idx].copy()
                self.P[u_idx] += self.lr * (err * self.Q[i_idx] - self.reg * self.P[u_idx])
                self.Q[i_idx] += self.lr * (err * p_old - self.reg * self.Q[i_idx])
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.mu
        u = self.user_to_idx[user_id]
        i = self.item_to_idx[item_id]
        pred = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
        return float(np.clip(pred, 0.5, 5.0))


class CollaborativeRecommender:
    """
    Production-grade Collaborative Filtering model using Surprise library
    with SVD (Matrix Factorization) and KNN (Memory-Based) algorithms.
    """
    def __init__(self, algorithm: str = "svd", n_factors: int = 50, n_epochs: int = 20, random_state: int = 42):
        self.algorithm_name = algorithm.lower()
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.random_state = random_state
        
        self.reader = Reader(rating_scale=(0.5, 5.0))
        self.model = None
        self.trainset = None
        self.global_mean: float = 3.5

    def fit(self, df_ratings: pd.DataFrame):
        """
        Fits SVD or KNN collaborative filtering model on user-item interaction records.
        """
        self.global_mean = float(df_ratings['rating'].mean())
        surprise_data = Dataset.load_from_df(df_ratings[['userId', 'movieId', 'rating']], self.reader)
        self.trainset = surprise_data.build_full_trainset()

        if self.algorithm_name == "knn":
            sim_options = {
                "name": "cosine",
                "user_based": True,
                "min_support": 3
            }
            self.model = KNNWithMeans(k=30, sim_options=sim_options, verbose=False)
        else:
            self.model = SVD(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
                lr_all=0.005,
                reg_all=0.02,
                random_state=self.random_state
            )

        self.model.fit(self.trainset)
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predicts the rating user_id would give to movie_id on a 0.5 - 5.0 scale.
        """
        if self.model is None:
            return self.global_mean
            
        try:
            pred = self.model.predict(uid=user_id, iid=movie_id)
            return float(np.clip(pred.est, 0.5, 5.0))
        except Exception:
            return self.global_mean

    def get_top_n_recommendations(
        self,
        user_id: int,
        catalog_movies: pd.DataFrame,
        rated_movie_ids: Optional[set] = None,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Scores all candidate unseen movies for a user and returns top N highest predicted titles.
        """
        rated_set = rated_movie_ids or set()
        candidates = catalog_movies[~catalog_movies['movieId'].isin(rated_set)]
        
        scored_movies = []
        for _, row in candidates.iterrows():
            mid = int(row['movieId'])
            pred_rating = self.predict_rating(user_id, mid)
            scored_movies.append({
                "movieId": mid,
                "title": row['clean_title'],
                "year": row['year'],
                "genres": row['genres'],
                "genres_list": row['genres_list'],
                "primary_genre": row['primary_genre'],
                "rating_mean": float(row['rating_mean']),
                "rating_count": int(row['rating_count']),
                "tmdbId": row.get('tmdbId'),
                "predicted_rating": round(pred_rating, 2),
                "explanation": f"Collaborative Filtering predicts you will rate this {pred_rating:.1f}/5.0 based on similar user preferences."
            })

        # Sort by predicted rating descending
        scored_movies.sort(key=lambda x: (x['predicted_rating'], x['rating_mean']), reverse=True)
        return scored_movies[:top_n]
