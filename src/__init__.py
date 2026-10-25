"""
CineMatch: A Hybrid Content-Based and Collaborative Filtering Movie Recommendation System
"""

from .data_loader import DataLoader, clean_movie_title
from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender, FunkSVDRecommender
from .hybrid import HybridRecommender
from .evaluation import ModelEvaluator
from .tmdb_client import TMDBClient

__all__ = [
    "DataLoader",
    "clean_movie_title",
    "ContentBasedRecommender",
    "CollaborativeRecommender",
    "FunkSVDRecommender",
    "HybridRecommender",
    "ModelEvaluator",
    "TMDBClient",
]
