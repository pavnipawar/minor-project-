"""
CineMatch: Hybrid Recommendation Engine
Combines Collaborative Filtering (SVD/KNN) and Content-Based Filtering (TF-IDF)
using an adaptive weighted scoring function with cold-start fallback and full explainability.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Set
from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender

class HybridRecommender:
    """
    Weighted Hybrid Recommender with adaptive cold-start fallback:
    Score(u, i) = alpha_eff * Norm(r_CF(u, i)) + (1 - alpha_eff) * Sim_CB(u, i)
    """
    def __init__(
        self,
        content_recommender: ContentBasedRecommender,
        collaborative_recommender: CollaborativeRecommender,
        default_alpha: float = 0.60,
        cold_start_threshold: int = 5
    ):
        self.cb = content_recommender
        self.cf = collaborative_recommender
        self.default_alpha = default_alpha
        self.cold_start_threshold = cold_start_threshold

    def calculate_effective_alpha(self, num_user_ratings: int, requested_alpha: Optional[float] = None) -> float:
        """
        Dynamically adjusts CF weight based on the volume of historical ratings available.
        For cold-start users (few or zero ratings), smoothly falls back to content-based filtering.
        """
        base_alpha = requested_alpha if requested_alpha is not None else self.default_alpha
        if num_user_ratings <= 0:
            return 0.0
        confidence_factor = min(1.0, num_user_ratings / float(self.cold_start_threshold))
        return base_alpha * confidence_factor

    def recommend_for_user(
        self,
        user_id: int,
        user_ratings_dict: Dict[int, float],
        catalog_movies: pd.DataFrame,
        top_n: int = 10,
        alpha: Optional[float] = None,
        candidate_pool_size: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Generates hybrid recommendations for a given user.
        Handles both existing dataset users and interactive cold-start new users.
        """
        rated_mids = set(user_ratings_dict.keys())
        num_ratings = len(rated_mids)
        eff_alpha = self.calculate_effective_alpha(num_ratings, alpha)
        eff_beta = 1.0 - eff_alpha

        # Select candidate items: exclude already rated, prioritize popular/relevant items for efficiency
        unrated_catalog = catalog_movies[~catalog_movies['movieId'].isin(rated_mids)]
        
        # Take top candidate pool sorted by bayesian rating & rating count to keep recommendation fast
        candidates = unrated_catalog.sort_values(by=['bayesian_rating', 'rating_count'], ascending=[False, False]).head(candidate_pool_size)

        # 1. Content-based profile similarities
        cb_scores_dict: Dict[int, float] = {}
        if eff_beta > 0.001 and num_ratings > 0:
            cb_recs = self.cb.get_recommendations_for_user_profile(user_ratings_dict, top_n=candidate_pool_size, exclude_rated=True)
            for rec in cb_recs:
                cb_scores_dict[rec['movieId']] = rec['similarity_score']

        # 2. Collaborative predictions and score combination
        hybrid_results = []
        is_cold_start = (num_ratings < self.cold_start_threshold)

        for _, row in candidates.iterrows():
            mid = int(row['movieId'])
            
            # Collaborative rating prediction (0.5 to 5.0)
            pred_cf = self.cf.predict_rating(user_id, mid)
            # Normalize CF to [0, 1] range: (rating - 0.5) / 4.5
            norm_cf = (pred_cf - 0.5) / 4.5
            
            # Content similarity [0, 1]
            sim_cb = cb_scores_dict.get(mid, 0.0)

            # Combined hybrid score [0, 1]
            hybrid_score = (eff_alpha * norm_cf) + (eff_beta * sim_cb)
            predicted_rating = 0.5 + (4.5 * hybrid_score)

            # Generate natural language explainability
            cf_pct = round(eff_alpha * 100)
            cb_pct = round(eff_beta * 100)
            
            if is_cold_start and eff_alpha == 0.0:
                explanation = f"Cold-Start Match: 100% matched against your favorite seed movies & genres ({row['primary_genre']})."
            elif is_cold_start:
                explanation = f"Hybrid Cold-Start: {cb_pct}% metadata taste match + {cf_pct}% collaborative signal."
            else:
                explanation = f"Hybrid Recommendation: {cf_pct}% taste pattern from similar users + {cb_pct}% content match on {row['primary_genre']} themes."

            hybrid_results.append({
                "movieId": mid,
                "title": row['clean_title'],
                "year": row['year'],
                "genres": row['genres'],
                "genres_list": row['genres_list'],
                "primary_genre": row['primary_genre'],
                "rating_mean": float(row['rating_mean']),
                "rating_count": int(row['rating_count']),
                "tmdbId": row.get('tmdbId'),
                "cf_predicted_rating": round(pred_cf, 2),
                "cb_similarity": round(sim_cb, 4),
                "hybrid_score": round(hybrid_score, 4),
                "predicted_rating": round(predicted_rating, 2),
                "match_pct": f"{min(99.5, max(60.0, round(hybrid_score * 100, 1))):.1f}%",
                "explanation": explanation,
                "is_cold_start": is_cold_start
            })

        # Sort by hybrid score descending
        hybrid_results.sort(key=lambda x: (x['hybrid_score'], x['rating_mean']), reverse=True)
        return hybrid_results[:top_n]
