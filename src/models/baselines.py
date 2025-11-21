"""Baseline recommendation models for comparison."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """Abstract base class for recommendation models."""
    
    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> None:
        """Fit the model to interactions data.
        
        Args:
            interactions: DataFrame with columns user_id, item_id, rating
        """
        pass
    
    @abstractmethod
    def recommend(
        self,
        user_id: int,
        candidate_items: Optional[List[int]] = None,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID
            candidate_items: Optional list of candidate items
            top_k: Number of recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        pass


class PopularityRecommender(BaseRecommender):
    """Popularity-based recommender."""
    
    def __init__(self) -> None:
        """Initialize popularity recommender."""
        self.item_popularity: Optional[np.ndarray] = None
        self.n_items: Optional[int] = None
    
    def fit(self, interactions: pd.DataFrame) -> None:
        """Fit popularity model.
        
        Args:
            interactions: DataFrame with columns user_id, item_id, rating
        """
        logger.info("Fitting popularity recommender...")
        
        # Calculate item popularity (average rating)
        item_stats = interactions.groupby('item_id')['rating'].agg(['mean', 'count'])
        self.item_popularity = item_stats['mean'].values
        self.n_items = len(self.item_popularity)
        
        logger.info(f"Fitted popularity model for {self.n_items} items")
    
    def recommend(
        self,
        user_id: int,
        candidate_items: Optional[List[int]] = None,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Generate popularity-based recommendations.
        
        Args:
            user_id: User ID (not used for popularity)
            candidate_items: Optional list of candidate items
            top_k: Number of recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        if candidate_items is None:
            candidate_items = list(range(self.n_items))
        
        # Get popularity scores for candidate items
        item_scores = [(item_id, self.item_popularity[item_id]) for item_id in candidate_items]
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        return item_scores[:top_k]
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating based on item popularity.
        
        Args:
            user_id: User ID (not used)
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        return float(self.item_popularity[item_id])


class UserKNNRecommender(BaseRecommender):
    """User-based collaborative filtering with k-nearest neighbors."""
    
    def __init__(self, k: int = 50, min_similarity: float = 0.1) -> None:
        """Initialize user KNN recommender.
        
        Args:
            k: Number of nearest neighbors
            min_similarity: Minimum similarity threshold
        """
        self.k = k
        self.min_similarity = min_similarity
        self.user_item_matrix: Optional[np.ndarray] = None
        self.user_similarities: Optional[np.ndarray] = None
        self.n_users: Optional[int] = None
        self.n_items: Optional[int] = None
    
    def fit(self, interactions: pd.DataFrame) -> None:
        """Fit user KNN model.
        
        Args:
            interactions: DataFrame with columns user_id, item_id, rating
        """
        logger.info(f"Fitting user KNN recommender with k={self.k}...")
        
        # Create user-item matrix
        self.n_users = interactions['user_id'].nunique()
        self.n_items = interactions['item_id'].nunique()
        
        self.user_item_matrix = np.zeros((self.n_users, self.n_items))
        
        for _, row in interactions.iterrows():
            self.user_item_matrix[row['user_id'], row['item_id']] = row['rating']
        
        # Calculate user similarities
        self.user_similarities = cosine_similarity(self.user_item_matrix)
        
        logger.info(f"Fitted user KNN model for {self.n_users} users and {self.n_items} items")
    
    def recommend(
        self,
        user_id: int,
        candidate_items: Optional[List[int]] = None,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Generate user KNN recommendations.
        
        Args:
            user_id: User ID
            candidate_items: Optional list of candidate items
            top_k: Number of recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        if candidate_items is None:
            candidate_items = list(range(self.n_items))
        
        # Get user's ratings
        user_ratings = self.user_item_matrix[user_id]
        
        # Find similar users
        user_similarities = self.user_similarities[user_id]
        similar_users = np.argsort(user_similarities)[::-1][1:self.k+1]  # Exclude self
        
        # Calculate predicted scores
        item_scores = []
        for item_id in candidate_items:
            if user_ratings[item_id] == 0:  # Only predict for unrated items
                score = self.predict(user_id, item_id)
                item_scores.append((item_id, score))
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:top_k]
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating using user KNN.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        # Get user's ratings
        user_ratings = self.user_item_matrix[user_id]
        
        # Find similar users who rated this item
        user_similarities = self.user_similarities[user_id]
        item_ratings = self.user_item_matrix[:, item_id]
        
        # Get users who rated this item
        rated_users = np.where(item_ratings > 0)[0]
        
        if len(rated_users) == 0:
            return 0.0
        
        # Calculate weighted average
        similarities = user_similarities[rated_users]
        ratings = item_ratings[rated_users]
        
        # Filter by minimum similarity
        valid_mask = similarities >= self.min_similarity
        if not np.any(valid_mask):
            return 0.0
        
        similarities = similarities[valid_mask]
        ratings = ratings[valid_mask]
        
        # Weighted average
        weighted_sum = np.sum(similarities * ratings)
        similarity_sum = np.sum(np.abs(similarities))
        
        if similarity_sum == 0:
            return 0.0
        
        return float(weighted_sum / similarity_sum)


class ItemKNNRecommender(BaseRecommender):
    """Item-based collaborative filtering with k-nearest neighbors."""
    
    def __init__(self, k: int = 50, min_similarity: float = 0.1) -> None:
        """Initialize item KNN recommender.
        
        Args:
            k: Number of nearest neighbors
            min_similarity: Minimum similarity threshold
        """
        self.k = k
        self.min_similarity = min_similarity
        self.user_item_matrix: Optional[np.ndarray] = None
        self.item_similarities: Optional[np.ndarray] = None
        self.n_users: Optional[int] = None
        self.n_items: Optional[int] = None
    
    def fit(self, interactions: pd.DataFrame) -> None:
        """Fit item KNN model.
        
        Args:
            interactions: DataFrame with columns user_id, item_id, rating
        """
        logger.info(f"Fitting item KNN recommender with k={self.k}...")
        
        # Create user-item matrix
        self.n_users = interactions['user_id'].nunique()
        self.n_items = interactions['item_id'].nunique()
        
        self.user_item_matrix = np.zeros((self.n_users, self.n_items))
        
        for _, row in interactions.iterrows():
            self.user_item_matrix[row['user_id'], row['item_id']] = row['rating']
        
        # Calculate item similarities
        self.item_similarities = cosine_similarity(self.user_item_matrix.T)
        
        logger.info(f"Fitted item KNN model for {self.n_users} users and {self.n_items} items")
    
    def recommend(
        self,
        user_id: int,
        candidate_items: Optional[List[int]] = None,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Generate item KNN recommendations.
        
        Args:
            user_id: User ID
            candidate_items: Optional list of candidate items
            top_k: Number of recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        if candidate_items is None:
            candidate_items = list(range(self.n_items))
        
        # Get user's ratings
        user_ratings = self.user_item_matrix[user_id]
        
        # Calculate predicted scores
        item_scores = []
        for item_id in candidate_items:
            if user_ratings[item_id] == 0:  # Only predict for unrated items
                score = self.predict(user_id, item_id)
                item_scores.append((item_id, score))
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:top_k]
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating using item KNN.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        # Get user's ratings
        user_ratings = self.user_item_matrix[user_id]
        
        # Find similar items that user rated
        item_similarities = self.item_similarities[item_id]
        
        # Get items user rated
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return 0.0
        
        # Calculate weighted average
        similarities = item_similarities[rated_items]
        ratings = user_ratings[rated_items]
        
        # Filter by minimum similarity
        valid_mask = similarities >= self.min_similarity
        if not np.any(valid_mask):
            return 0.0
        
        similarities = similarities[valid_mask]
        ratings = ratings[valid_mask]
        
        # Weighted average
        weighted_sum = np.sum(similarities * ratings)
        similarity_sum = np.sum(np.abs(similarities))
        
        if similarity_sum == 0:
            return 0.0
        
        return float(weighted_sum / similarity_sum)


class MatrixFactorizationRecommender(BaseRecommender):
    """Matrix Factorization using Alternating Least Squares."""
    
    def __init__(
        self,
        factors: int = 50,
        iterations: int = 100,
        regularization: float = 0.01,
        learning_rate: float = 0.01
    ) -> None:
        """Initialize matrix factorization recommender.
        
        Args:
            factors: Number of latent factors
            iterations: Number of iterations
            regularization: Regularization parameter
            learning_rate: Learning rate
        """
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.learning_rate = learning_rate
        
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self.global_bias: Optional[float] = None
        
        self.n_users: Optional[int] = None
        self.n_items: Optional[int] = None
    
    def fit(self, interactions: pd.DataFrame) -> None:
        """Fit matrix factorization model.
        
        Args:
            interactions: DataFrame with columns user_id, item_id, rating
        """
        logger.info(f"Fitting matrix factorization with {self.factors} factors...")
        
        self.n_users = interactions['user_id'].nunique()
        self.n_items = interactions['item_id'].nunique()
        
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.factors))
        
        # Initialize biases
        self.global_bias = interactions['rating'].mean()
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        
        # Convert to numpy arrays for efficiency
        user_ids = interactions['user_id'].values
        item_ids = interactions['item_id'].values
        ratings = interactions['rating'].values
        
        # Training loop
        for iteration in range(self.iterations):
            # Update user factors
            for u in range(self.n_users):
                user_mask = user_ids == u
                if not np.any(user_mask):
                    continue
                
                user_items = item_ids[user_mask]
                user_ratings = ratings[user_mask]
                
                # Calculate error
                predicted = (
                    self.global_bias + 
                    self.user_bias[u] + 
                    self.item_bias[user_items] + 
                    np.sum(self.user_factors[u] * self.item_factors[user_items], axis=1)
                )
                error = user_ratings - predicted
                
                # Update user factors
                item_factor_sum = np.sum(self.item_factors[user_items], axis=0)
                self.user_factors[u] = (
                    self.user_factors[u] + 
                    self.learning_rate * (np.sum(error[:, np.newaxis] * self.item_factors[user_items], axis=0) - 
                                         self.regularization * self.user_factors[u])
                )
                
                # Update user bias
                self.user_bias[u] += self.learning_rate * (np.mean(error) - self.regularization * self.user_bias[u])
            
            # Update item factors
            for i in range(self.n_items):
                item_mask = item_ids == i
                if not np.any(item_mask):
                    continue
                
                item_users = user_ids[item_mask]
                item_ratings = ratings[item_mask]
                
                # Calculate error
                predicted = (
                    self.global_bias + 
                    self.user_bias[item_users] + 
                    self.item_bias[i] + 
                    np.sum(self.user_factors[item_users] * self.item_factors[i], axis=1)
                )
                error = item_ratings - predicted
                
                # Update item factors
                user_factor_sum = np.sum(self.user_factors[item_users], axis=0)
                self.item_factors[i] = (
                    self.item_factors[i] + 
                    self.learning_rate * (np.sum(error[:, np.newaxis] * self.user_factors[item_users], axis=0) - 
                                         self.regularization * self.item_factors[i])
                )
                
                # Update item bias
                self.item_bias[i] += self.learning_rate * (np.mean(error) - self.regularization * self.item_bias[i])
            
            if iteration % 20 == 0:
                logger.debug(f"Matrix factorization iteration {iteration}")
        
        logger.info("Matrix factorization training completed")
    
    def recommend(
        self,
        user_id: int,
        candidate_items: Optional[List[int]] = None,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Generate matrix factorization recommendations.
        
        Args:
            user_id: User ID
            candidate_items: Optional list of candidate items
            top_k: Number of recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        if candidate_items is None:
            candidate_items = list(range(self.n_items))
        
        # Calculate predicted scores
        item_scores = []
        for item_id in candidate_items:
            score = self.predict(user_id, item_id)
            item_scores.append((item_id, score))
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:top_k]
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating using matrix factorization.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        prediction = (
            self.global_bias + 
            self.user_bias[user_id] + 
            self.item_bias[item_id] + 
            np.dot(self.user_factors[user_id], self.item_factors[item_id])
        )
        return float(prediction)


def create_baseline_model(model_type: str, **kwargs) -> BaseRecommender:
    """Create baseline model by type.
    
    Args:
        model_type: Type of baseline model
        **kwargs: Model-specific parameters
        
    Returns:
        Baseline recommender instance
    """
    if model_type == "popularity":
        return PopularityRecommender()
    elif model_type == "user_knn":
        return UserKNNRecommender(**kwargs)
    elif model_type == "item_knn":
        return ItemKNNRecommender(**kwargs)
    elif model_type == "matrix_factorization":
        return MatrixFactorizationRecommender(**kwargs)
    else:
        raise ValueError(f"Unknown baseline model type: {model_type}")
