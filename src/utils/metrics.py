"""Evaluation metrics for recommendation systems."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


class RecommendationMetrics:
    """Collection of recommendation evaluation metrics."""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]) -> None:
        """Initialize metrics calculator.
        
        Args:
            k_values: List of k values for ranking metrics
        """
        self.k_values = k_values
    
    def precision_at_k(
        self,
        recommendations: List[int],
        relevant_items: List[int],
        k: int
    ) -> float:
        """Calculate Precision@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        if len(top_k_recs) == 0:
            return 0.0
        
        relevant_recommendations = sum(1 for item in top_k_recs if item in relevant_set)
        return relevant_recommendations / len(top_k_recs)
    
    def recall_at_k(
        self,
        recommendations: List[int],
        relevant_items: List[int],
        k: int
    ) -> float:
        """Calculate Recall@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        relevant_recommendations = sum(1 for item in top_k_recs if item in relevant_set)
        return relevant_recommendations / len(relevant_items)
    
    def ndcg_at_k(
        self,
        recommendations: List[int],
        relevant_items: List[int],
        k: int
    ) -> float:
        """Calculate NDCG@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_recs):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(len(relevant_items), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def map_at_k(
        self,
        recommendations: List[int],
        relevant_items: List[int],
        k: int
    ) -> float:
        """Calculate MAP@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            MAP@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_recs):
            if item in relevant_set:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        if relevant_count == 0:
            return 0.0
        
        return precision_sum / len(relevant_items)
    
    def hit_rate_at_k(
        self,
        recommendations: List[int],
        relevant_items: List[int],
        k: int
    ) -> float:
        """Calculate Hit Rate@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Hit Rate@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        return 1.0 if any(item in relevant_set for item in top_k_recs) else 0.0
    
    def coverage(
        self,
        recommendations: Dict[int, List[int]],
        all_items: List[int]
    ) -> float:
        """Calculate catalog coverage.
        
        Args:
            recommendations: Dictionary mapping user_id to recommendations
            all_items: List of all available items
            
        Returns:
            Coverage score
        """
        if not recommendations:
            return 0.0
        
        recommended_items = set()
        for user_recs in recommendations.values():
            recommended_items.update(user_recs)
        
        return len(recommended_items) / len(all_items)
    
    def diversity(
        self,
        recommendations: Dict[int, List[int]],
        item_features: Optional[pd.DataFrame] = None
    ) -> float:
        """Calculate diversity of recommendations.
        
        Args:
            recommendations: Dictionary mapping user_id to recommendations
            item_features: Optional DataFrame with item features
            
        Returns:
            Diversity score
        """
        if not recommendations:
            return 0.0
        
        # Simple diversity: average pairwise Jaccard distance
        diversity_scores = []
        
        for user_id, user_recs in recommendations.items():
            if len(user_recs) < 2:
                continue
            
            # Calculate pairwise Jaccard distances
            jaccard_distances = []
            for i in range(len(user_recs)):
                for j in range(i + 1, len(user_recs)):
                    item1, item2 = user_recs[i], user_recs[j]
                    
                    if item_features is not None and 'category' in item_features.columns:
                        # Use categorical features for diversity
                        cat1 = item_features.loc[item1, 'category'] if item1 in item_features.index else 'unknown'
                        cat2 = item_features.loc[item2, 'category'] if item2 in item_features.index else 'unknown'
                        jaccard_dist = 1.0 if cat1 != cat2 else 0.0
                    else:
                        # Simple binary diversity
                        jaccard_dist = 1.0 if item1 != item2 else 0.0
                    
                    jaccard_distances.append(jaccard_dist)
            
            if jaccard_distances:
                diversity_scores.append(np.mean(jaccard_distances))
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def novelty(
        self,
        recommendations: Dict[int, List[int]],
        item_popularity: Dict[int, float]
    ) -> float:
        """Calculate novelty of recommendations.
        
        Args:
            recommendations: Dictionary mapping user_id to recommendations
            item_popularity: Dictionary mapping item_id to popularity score
            
        Returns:
            Novelty score
        """
        if not recommendations:
            return 0.0
        
        novelty_scores = []
        
        for user_recs in recommendations.values():
            user_novelty = []
            for item_id in user_recs:
                # Novelty is inverse of popularity
                popularity = item_popularity.get(item_id, 0.0)
                novelty = 1.0 - popularity
                user_novelty.append(novelty)
            
            if user_novelty:
                novelty_scores.append(np.mean(user_novelty))
        
        return np.mean(novelty_scores) if novelty_scores else 0.0
    
    def calculate_ranking_metrics(
        self,
        recommendations: Dict[int, List[int]],
        test_interactions: pd.DataFrame,
        k_values: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """Calculate all ranking metrics.
        
        Args:
            recommendations: Dictionary mapping user_id to recommendations
            test_interactions: Test interactions DataFrame
            k_values: Optional list of k values
            
        Returns:
            Dictionary of metric scores
        """
        if k_values is None:
            k_values = self.k_values
        
        metrics = {}
        
        # Group test interactions by user
        user_relevant_items = {}
        for user_id in test_interactions['user_id'].unique():
            user_items = test_interactions[test_interactions['user_id'] == user_id]['item_id'].tolist()
            user_relevant_items[int(user_id)] = [int(item_id) for item_id in user_items]
        
        for k in k_values:
            precisions = []
            recalls = []
            ndcgs = []
            maps = []
            hit_rates = []
            
            for user_id, user_recs in recommendations.items():
                if user_id not in user_relevant_items:
                    continue
                
                relevant_items = user_relevant_items[user_id]
                
                precisions.append(self.precision_at_k(user_recs, relevant_items, k))
                recalls.append(self.recall_at_k(user_recs, relevant_items, k))
                ndcgs.append(self.ndcg_at_k(user_recs, relevant_items, k))
                maps.append(self.map_at_k(user_recs, relevant_items, k))
                hit_rates.append(self.hit_rate_at_k(user_recs, relevant_items, k))
            
            metrics[f'precision@{k}'] = np.mean(precisions) if precisions else 0.0
            metrics[f'recall@{k}'] = np.mean(recalls) if recalls else 0.0
            metrics[f'ndcg@{k}'] = np.mean(ndcgs) if ndcgs else 0.0
            metrics[f'map@{k}'] = np.mean(maps) if maps else 0.0
            metrics[f'hit_rate@{k}'] = np.mean(hit_rates) if hit_rates else 0.0
        
        return metrics
    
    def calculate_rating_metrics(
        self,
        predictions: Dict[Tuple[int, int], float],
        test_interactions: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate rating prediction metrics.
        
        Args:
            predictions: Dictionary mapping (user_id, item_id) to predicted rating
            test_interactions: Test interactions DataFrame
            
        Returns:
            Dictionary of metric scores
        """
        if not predictions:
            return {'mse': 0.0, 'rmse': 0.0, 'mae': 0.0}
        
        # Get actual ratings
        actual_ratings = []
        predicted_ratings = []
        
        for _, row in test_interactions.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            if (user_id, item_id) in predictions:
                predicted_rating = predictions[(user_id, item_id)]
                actual_ratings.append(actual_rating)
                predicted_ratings.append(predicted_rating)
        
        if not actual_ratings:
            return {'mse': 0.0, 'rmse': 0.0, 'mae': 0.0}
        
        actual_ratings = np.array(actual_ratings)
        predicted_ratings = np.array(predicted_ratings)
        
        mse = mean_squared_error(actual_ratings, predicted_ratings)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_ratings, predicted_ratings)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
    
    def calculate_all_metrics(
        self,
        recommendations: Dict[int, List[int]],
        predictions: Dict[Tuple[int, int], float],
        test_interactions: pd.DataFrame,
        all_items: List[int],
        item_features: Optional[pd.DataFrame] = None,
        item_popularity: Optional[Dict[int, float]] = None
    ) -> Dict[str, float]:
        """Calculate all metrics.
        
        Args:
            recommendations: Dictionary mapping user_id to recommendations
            predictions: Dictionary mapping (user_id, item_id) to predicted rating
            test_interactions: Test interactions DataFrame
            all_items: List of all available items
            item_features: Optional DataFrame with item features
            item_popularity: Optional dictionary mapping item_id to popularity
            
        Returns:
            Dictionary of all metric scores
        """
        metrics = {}
        
        # Ranking metrics
        ranking_metrics = self.calculate_ranking_metrics(recommendations, test_interactions)
        metrics.update(ranking_metrics)
        
        # Rating metrics
        rating_metrics = self.calculate_rating_metrics(predictions, test_interactions)
        metrics.update(rating_metrics)
        
        # Coverage
        metrics['coverage'] = self.coverage(recommendations, all_items)
        
        # Diversity
        metrics['diversity'] = self.diversity(recommendations, item_features)
        
        # Novelty
        if item_popularity is not None:
            metrics['novelty'] = self.novelty(recommendations, item_popularity)
        
        return metrics


def evaluate_model(
    model,
    test_interactions: pd.DataFrame,
    all_items: List[int],
    item_features: Optional[pd.DataFrame] = None,
    k_values: List[int] = [5, 10, 20],
    top_k: int = 10
) -> Dict[str, float]:
    """Evaluate a recommendation model.
    
    Args:
        model: Recommendation model
        test_interactions: Test interactions DataFrame
        all_items: List of all available items
        item_features: Optional DataFrame with item features
        k_values: List of k values for ranking metrics
        top_k: Number of recommendations to generate
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model...")
    
    # Generate recommendations for all users in test set
    recommendations = {}
    predictions = {}
    
    for user_id in test_interactions['user_id'].unique():
        try:
            # Generate recommendations
            user_recs = model.recommend(int(user_id), top_k=top_k)
            recommendations[int(user_id)] = [item_id for item_id, _ in user_recs]
            
            # Generate predictions for test interactions
            user_test_items = test_interactions[test_interactions['user_id'] == user_id]['item_id'].tolist()
            for item_id in user_test_items:
                try:
                    pred_rating = model.predict(int(user_id), int(item_id))
                    predictions[(int(user_id), int(item_id))] = pred_rating
                except Exception as e:
                    logger.warning(f"Failed to predict for user {user_id}, item {item_id}: {e}")
                    predictions[(int(user_id), int(item_id))] = 0.0
        
        except Exception as e:
            logger.warning(f"Failed to generate recommendations for user {user_id}: {e}")
            recommendations[int(user_id)] = []
    
    # Calculate metrics
    metrics_calculator = RecommendationMetrics(k_values=k_values)
    
    # Calculate item popularity for novelty metric
    item_popularity = None
    if len(test_interactions) > 0:
        item_counts = test_interactions['item_id'].value_counts()
        item_popularity = {item_id: count / len(test_interactions) for item_id, count in item_counts.items()}
    
    metrics = metrics_calculator.calculate_all_metrics(
        recommendations=recommendations,
        predictions=predictions,
        test_interactions=test_interactions,
        all_items=all_items,
        item_features=item_features,
        item_popularity=item_popularity
    )
    
    logger.info("Model evaluation completed")
    return metrics
