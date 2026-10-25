"""
CineMatch: Model Evaluation Engine
Performs rigorous, leak-free train/test evaluation across Content-Based, SVD, KNN,
and Hybrid models, calculating RMSE, MAE, Precision@K, Recall@K, and Catalog Coverage.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from surprise import SVD, KNNWithMeans, Reader, Dataset
from .content_based import ContentBasedRecommender
from .data_loader import DataLoader

class ModelEvaluator:
    """
    Evaluates recommender algorithms on held-out test data without data leakage.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.results_path = os.path.join(data_dir, "evaluation_results.json")

    def run_evaluation(
        self,
        dl: DataLoader,
        test_size: float = 0.2,
        k_values: List[int] = [5, 10, 20],
        relevance_threshold: float = 3.5,
        user_sample_size: int = 120,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Executes complete comparative benchmark on held-out test split.
        """
        t_start = time.time()
        df_ratings = dl.df_ratings.copy()
        
        # 1. Held-out train-test split strictly preventing data leakage
        train_df, test_df = train_test_split(df_ratings, test_size=test_size, random_state=random_state)
        
        # Precompute user training profiles
        user_train_ratings: Dict[int, Dict[int, float]] = {}
        user_train_means: Dict[int, float] = {}
        for uid, group in train_df.groupby('userId'):
            user_train_ratings[uid] = dict(zip(group['movieId'], group['rating']))
            user_train_means[uid] = float(group['rating'].mean())
            
        global_mean = float(train_df['rating'].mean())

        # 2. Train Content-Based Recommender (using movie catalog metadata)
        cb = ContentBasedRecommender(max_features=8000)
        cb.fit(dl.df_movies)

        # 3. Train Collaborative SVD
        reader = Reader(rating_scale=(0.5, 5.0))
        data_train = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
        trainset = data_train.build_full_trainset()
        
        svd_model = SVD(n_factors=40, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=random_state)
        svd_model.fit(trainset)

        # 4. Train Collaborative KNN (WithMeans)
        knn_model = KNNWithMeans(k=30, sim_options={'name': 'cosine', 'user_based': True}, verbose=False)
        knn_model.fit(trainset)

        # 5. Evaluate Rating Prediction Accuracy (RMSE & MAE)
        test_pairs = list(zip(test_df['userId'], test_df['movieId'], test_df['rating']))
        
        # Limit prediction sample to 10,000 pairs for snappy interactive speed if large
        eval_pairs = test_pairs[:10000] if len(test_pairs) > 10000 else test_pairs

        svd_errors, knn_errors, cb_errors, hybrid_errors = [], [], [], []
        svd_abs_err, knn_abs_err, cb_abs_err, hybrid_abs_err = [], [], [], []

        # Precompute movie vectors for fast similarity
        for uid, mid, true_r in eval_pairs:
            # SVD prediction
            p_svd = float(np.clip(svd_model.predict(uid, mid).est, 0.5, 5.0))
            
            # KNN prediction
            p_knn = float(np.clip(knn_model.predict(uid, mid).est, 0.5, 5.0))

            # Content-Based prediction
            u_ratings = user_train_ratings.get(uid, {})
            u_mean = user_train_means.get(uid, global_mean)
            
            if u_ratings:
                # Find similarity between mid and user liked movies
                sims = [cb.get_similarity_score(mid, liked_m) for liked_m, r in u_ratings.items() if r >= 3.5]
                avg_sim = np.mean(sims) if sims else 0.0
                p_cb = float(np.clip(u_mean + (avg_sim - 0.2) * 2.5, 0.5, 5.0))
            else:
                p_cb = u_mean

            # Hybrid prediction (alpha = 0.6 CF + 0.4 CB)
            p_hybrid = float(np.clip((0.65 * p_svd) + (0.35 * p_cb), 0.5, 5.0))

            # Collect errors
            svd_errors.append((true_r - p_svd) ** 2)
            knn_errors.append((true_r - p_knn) ** 2)
            cb_errors.append((true_r - p_cb) ** 2)
            hybrid_errors.append((true_r - p_hybrid) ** 2)

            svd_abs_err.append(abs(true_r - p_svd))
            knn_abs_err.append(abs(true_r - p_knn))
            cb_abs_err.append(abs(true_r - p_cb))
            hybrid_abs_err.append(abs(true_r - p_hybrid))

        rmse_svd = float(np.sqrt(np.mean(svd_errors)))
        mae_svd = float(np.mean(svd_abs_err))

        rmse_knn = float(np.sqrt(np.mean(knn_errors)))
        mae_knn = float(np.mean(knn_abs_err))

        rmse_cb = float(np.sqrt(np.mean(cb_errors)))
        mae_cb = float(np.mean(cb_abs_err))

        rmse_hybrid = float(np.sqrt(np.mean(hybrid_errors)))
        mae_hybrid = float(np.mean(hybrid_abs_err))

        # 6. Evaluate Ranking Metrics: Precision@K & Recall@K
        # Select test users with at least 5 test ratings
        test_user_counts = test_df['userId'].value_counts()
        eligible_users = test_user_counts[test_user_counts >= 5].index.tolist()
        
        # Sample users for ranking metrics evaluation
        np.random.seed(random_state)
        sample_users = np.random.choice(eligible_users, size=min(user_sample_size, len(eligible_users)), replace=False)

        metrics = {
            "SVD": {f"p_{k}": [] for k in k_values} | {f"r_{k}": [] for k in k_values} | {"rec_items": set()},
            "KNN": {f"p_{k}": [] for k in k_values} | {f"r_{k}": [] for k in k_values} | {"rec_items": set()},
            "Content-Based": {f"p_{k}": [] for k in k_values} | {f"r_{k}": [] for k in k_values} | {"rec_items": set()},
            "Hybrid": {f"p_{k}": [] for k in k_values} | {f"r_{k}": [] for k in k_values} | {"rec_items": set()}
        }

        max_k = max(k_values)

        for uid in sample_users:
            u_test = test_df[test_df['userId'] == uid]
            relevant_items = set(u_test[u_test['rating'] >= relevance_threshold]['movieId'])
            if not relevant_items:
                continue

            test_item_pool = u_test['movieId'].tolist()
            u_train = user_train_ratings.get(uid, {})
            u_mean = user_train_means.get(uid, global_mean)

            # Score test items across all 4 models
            scores_svd, scores_knn, scores_cb, scores_hyb = [], [], [], []
            for mid in test_item_pool:
                # SVD
                s_svd = svd_model.predict(uid, mid).est
                scores_svd.append((mid, s_svd))
                # KNN
                s_knn = knn_model.predict(uid, mid).est
                scores_knn.append((mid, s_knn))
                # CB
                sims = [cb.get_similarity_score(mid, lm) for lm, r in u_train.items() if r >= 3.5]
                s_cb = np.mean(sims) if sims else 0.0
                scores_cb.append((mid, s_cb))
                # Hybrid
                s_hyb = (0.6 * ((s_svd - 0.5) / 4.5)) + (0.4 * s_cb)
                scores_hyb.append((mid, s_hyb))

            model_rankings = {
                "SVD": [m for m, _ in sorted(scores_svd, key=lambda x: x[1], reverse=True)],
                "KNN": [m for m, _ in sorted(scores_knn, key=lambda x: x[1], reverse=True)],
                "Content-Based": [m for m, _ in sorted(scores_cb, key=lambda x: x[1], reverse=True)],
                "Hybrid": [m for m, _ in sorted(scores_hyb, key=lambda x: x[1], reverse=True)]
            }

            for model_name, ranked in model_rankings.items():
                for k in k_values:
                    top_k = ranked[:k]
                    n_rel_in_k = len(set(top_k).intersection(relevant_items))
                    precision = n_rel_in_k / float(k)
                    recall = n_rel_in_k / float(len(relevant_items))
                    metrics[model_name][f"p_{k}"].append(precision)
                    metrics[model_name][f"r_{k}"].append(recall)
                metrics[model_name]["rec_items"].update(ranked[:max_k])

        total_catalog_items = len(dl.df_movies)

        results = {
            "metadata": {
                "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_ratings": len(df_ratings),
                "train_ratings": len(train_df),
                "test_ratings": len(test_df),
                "evaluation_duration_seconds": round(time.time() - t_start, 2),
                "evaluated_users_count": len(sample_users),
                "relevance_threshold": relevance_threshold
            },
            "models": {
                "Collaborative (SVD)": {
                    "rmse": round(rmse_svd, 4),
                    "mae": round(mae_svd, 4),
                    "precision_5": round(float(np.mean(metrics["SVD"]["p_5"])), 4),
                    "precision_10": round(float(np.mean(metrics["SVD"]["p_10"])), 4),
                    "precision_20": round(float(np.mean(metrics["SVD"]["p_20"])), 4),
                    "recall_5": round(float(np.mean(metrics["SVD"]["r_5"])), 4),
                    "recall_10": round(float(np.mean(metrics["SVD"]["r_10"])), 4),
                    "recall_20": round(float(np.mean(metrics["SVD"]["r_20"])), 4),
                    "coverage_pct": round(len(metrics["SVD"]["rec_items"]) / total_catalog_items * 100, 2),
                    "type": "Collaborative (Matrix Factorization)",
                    "best_for": "Latent taste pattern discovery"
                },
                "Collaborative (KNN)": {
                    "rmse": round(rmse_knn, 4),
                    "mae": round(mae_knn, 4),
                    "precision_5": round(float(np.mean(metrics["KNN"]["p_5"])), 4),
                    "precision_10": round(float(np.mean(metrics["KNN"]["p_10"])), 4),
                    "precision_20": round(float(np.mean(metrics["KNN"]["p_20"])), 4),
                    "recall_5": round(float(np.mean(metrics["KNN"]["r_5"])), 4),
                    "recall_10": round(float(np.mean(metrics["KNN"]["r_10"])), 4),
                    "recall_20": round(float(np.mean(metrics["KNN"]["r_20"])), 4),
                    "coverage_pct": round(len(metrics["KNN"]["rec_items"]) / total_catalog_items * 100, 2),
                    "type": "Collaborative (Memory-Based KNN)",
                    "best_for": "Interpretable peer similarity"
                },
                "Content-Based (TF-IDF)": {
                    "rmse": round(rmse_cb, 4),
                    "mae": round(mae_cb, 4),
                    "precision_5": round(float(np.mean(metrics["Content-Based"]["p_5"])), 4),
                    "precision_10": round(float(np.mean(metrics["Content-Based"]["p_10"])), 4),
                    "precision_20": round(float(np.mean(metrics["Content-Based"]["p_20"])), 4),
                    "recall_5": round(float(np.mean(metrics["Content-Based"]["r_5"])), 4),
                    "recall_10": round(float(np.mean(metrics["Content-Based"]["r_10"])), 4),
                    "recall_20": round(float(np.mean(metrics["Content-Based"]["r_20"])), 4),
                    "coverage_pct": round(len(metrics["Content-Based"]["rec_items"]) / total_catalog_items * 100, 2),
                    "type": "Content-Based (TF-IDF Cosine)",
                    "best_for": "Cold-start resilience & thematic fidelity"
                },
                "CineMatch Hybrid": {
                    "rmse": round(rmse_hybrid, 4),
                    "mae": round(mae_hybrid, 4),
                    "precision_5": round(float(np.mean(metrics["Hybrid"]["p_5"])), 4),
                    "precision_10": round(float(np.mean(metrics["Hybrid"]["p_10"])), 4),
                    "precision_20": round(float(np.mean(metrics["Hybrid"]["p_20"])), 4),
                    "recall_5": round(float(np.mean(metrics["Hybrid"]["r_5"])), 4),
                    "recall_10": round(float(np.mean(metrics["Hybrid"]["r_10"])), 4),
                    "recall_20": round(float(np.mean(metrics["Hybrid"]["r_20"])), 4),
                    "coverage_pct": round(len(metrics["Hybrid"]["rec_items"]) / total_catalog_items * 100, 2),
                    "type": "Hybrid (Weighted Score + Cold-Start Fallback)",
                    "best_for": "Optimal balance of accuracy, diversity & cold-start"
                }
            }
        }

        # Save to disk cache
        try:
            with open(self.results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception:
            pass

        return results

    def load_or_run(self, dl: DataLoader) -> Dict[str, Any]:
        """Loads cached evaluation results if available, else computes them."""
        if os.path.exists(self.results_path):
            try:
                with open(self.results_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return self.run_evaluation(dl)
