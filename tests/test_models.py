"""Unit tests for Wide & Deep Learning Recommendation System."""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch

# Import modules to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.utils import DataGenerator, DataLoader, set_seed
from src.models.wide_deep import WideAndDeepModel, WideAndDeepTrainer, RecommendationDataset
from src.models.baselines import (
    PopularityRecommender, 
    UserKNNRecommender, 
    ItemKNNRecommender, 
    MatrixFactorizationRecommender,
    create_baseline_model
)
from src.utils.metrics import RecommendationMetrics


class TestDataGenerator:
    """Test DataGenerator class."""
    
    def test_init(self):
        """Test DataGenerator initialization."""
        generator = DataGenerator(n_users=100, n_items=50, n_interactions=1000)
        assert generator.n_users == 100
        assert generator.n_items == 50
        assert generator.n_interactions == 1000
    
    def test_generate_interactions(self):
        """Test interaction generation."""
        generator = DataGenerator(n_users=10, n_items=5, n_interactions=20, random_state=42)
        interactions = generator.generate_interactions()
        
        assert len(interactions) == 20
        assert 'user_id' in interactions.columns
        assert 'item_id' in interactions.columns
        assert 'rating' in interactions.columns
        assert 'timestamp' in interactions.columns
        assert interactions['user_id'].min() >= 0
        assert interactions['user_id'].max() < 10
        assert interactions['item_id'].min() >= 0
        assert interactions['item_id'].max() < 5
    
    def test_generate_items(self):
        """Test item generation."""
        generator = DataGenerator(n_items=10, random_state=42)
        items = generator.generate_items()
        
        assert len(items) == 10
        assert 'item_id' in items.columns
        assert 'title' in items.columns
        assert 'category' in items.columns
        assert 'price' in items.columns
        assert 'description' in items.columns
    
    def test_generate_users(self):
        """Test user generation."""
        generator = DataGenerator(n_users=10, random_state=42)
        users = generator.generate_users()
        
        assert len(users) == 10
        assert 'user_id' in users.columns
        assert 'age' in users.columns
        assert 'gender' in users.columns
        assert 'location' in users.columns
        assert 'registration_date' in users.columns


class TestDataLoader:
    """Test DataLoader class."""
    
    def test_preprocess_interactions(self):
        """Test interaction preprocessing."""
        # Create test data
        interactions = pd.DataFrame({
            'user_id': [0, 0, 1, 1, 2, 2, 2],
            'item_id': [0, 1, 0, 2, 0, 1, 2],
            'rating': [5, 4, 3, 2, 1, 5, 4]
        })
        
        data_loader = DataLoader(".")
        processed = data_loader.preprocess_interactions(interactions, min_user_interactions=2, min_item_interactions=2)
        
        # Should filter out user 0 and item 0 (not enough interactions)
        assert len(processed) == 3  # Only user 2's interactions remain
        assert processed['user_id'].nunique() == 1
        assert processed['item_id'].nunique() == 2
    
    def test_split_data(self):
        """Test data splitting."""
        interactions = pd.DataFrame({
            'user_id': [0, 0, 1, 1, 2, 2, 3, 3],
            'item_id': [0, 1, 0, 2, 0, 1, 0, 2],
            'rating': [5, 4, 3, 2, 1, 5, 4, 3]
        })
        
        data_loader = DataLoader(".")
        train_df, val_df, test_df = data_loader.split_data(interactions, test_size=0.25, val_size=0.25)
        
        assert len(train_df) + len(val_df) + len(test_df) == len(interactions)
        assert len(test_df) == 2  # 25% of 8 = 2
        assert len(val_df) == 2   # 25% of remaining 6 = 2
        assert len(train_df) == 4  # Remaining


class TestWideAndDeepModel:
    """Test WideAndDeepModel class."""
    
    def test_init(self):
        """Test model initialization."""
        model = WideAndDeepModel(n_users=100, n_items=50, embedding_dim=32)
        assert model.n_users == 100
        assert model.n_items == 50
        assert model.embedding_dim == 32
    
    def test_forward(self):
        """Test forward pass."""
        model = WideAndDeepModel(n_users=10, n_items=5, embedding_dim=16)
        
        user_ids = torch.tensor([0, 1, 2])
        item_ids = torch.tensor([0, 1, 2])
        
        output = model(user_ids, item_ids)
        
        assert output.shape == (3,)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestRecommendationDataset:
    """Test RecommendationDataset class."""
    
    def test_init(self):
        """Test dataset initialization."""
        interactions = np.array([[0, 1], [1, 2], [2, 0]])
        ratings = np.array([5.0, 4.0, 3.0])
        
        dataset = RecommendationDataset(interactions, ratings)
        assert len(dataset) == 3
    
    def test_getitem(self):
        """Test dataset indexing."""
        interactions = np.array([[0, 1], [1, 2]])
        ratings = np.array([5.0, 4.0])
        
        dataset = RecommendationDataset(interactions, ratings)
        
        user_id, item_id, rating = dataset[0]
        assert user_id.item() == 0
        assert item_id.item() == 1
        assert rating.item() == 5.0


class TestBaselineModels:
    """Test baseline recommendation models."""
    
    def test_popularity_recommender(self):
        """Test PopularityRecommender."""
        interactions = pd.DataFrame({
            'user_id': [0, 0, 1, 1, 2, 2],
            'item_id': [0, 1, 0, 2, 0, 1],
            'rating': [5, 4, 3, 2, 1, 5]
        })
        
        model = PopularityRecommender()
        model.fit(interactions)
        
        recommendations = model.recommend(0, top_k=2)
        assert len(recommendations) == 2
        assert all(isinstance(item_id, int) for item_id, _ in recommendations)
        assert all(isinstance(score, float) for _, score in recommendations)
    
    def test_user_knn_recommender(self):
        """Test UserKNNRecommender."""
        interactions = pd.DataFrame({
            'user_id': [0, 0, 1, 1, 2, 2],
            'item_id': [0, 1, 0, 2, 0, 1],
            'rating': [5, 4, 3, 2, 1, 5]
        })
        
        model = UserKNNRecommender(k=2)
        model.fit(interactions)
        
        recommendations = model.recommend(0, top_k=2)
        assert len(recommendations) <= 2
    
    def test_item_knn_recommender(self):
        """Test ItemKNNRecommender."""
        interactions = pd.DataFrame({
            'user_id': [0, 0, 1, 1, 2, 2],
            'item_id': [0, 1, 0, 2, 0, 1],
            'rating': [5, 4, 3, 2, 1, 5]
        })
        
        model = ItemKNNRecommender(k=2)
        model.fit(interactions)
        
        recommendations = model.recommend(0, top_k=2)
        assert len(recommendations) <= 2
    
    def test_matrix_factorization_recommender(self):
        """Test MatrixFactorizationRecommender."""
        interactions = pd.DataFrame({
            'user_id': [0, 0, 1, 1, 2, 2],
            'item_id': [0, 1, 0, 2, 0, 1],
            'rating': [5, 4, 3, 2, 1, 5]
        })
        
        model = MatrixFactorizationRecommender(factors=5, iterations=10)
        model.fit(interactions)
        
        recommendations = model.recommend(0, top_k=2)
        assert len(recommendations) <= 2
    
    def test_create_baseline_model(self):
        """Test create_baseline_model function."""
        # Test popularity model
        model = create_baseline_model("popularity")
        assert isinstance(model, PopularityRecommender)
        
        # Test user_knn model
        model = create_baseline_model("user_knn", k=10)
        assert isinstance(model, UserKNNRecommender)
        assert model.k == 10
        
        # Test invalid model type
        with pytest.raises(ValueError):
            create_baseline_model("invalid_model")


class TestRecommendationMetrics:
    """Test RecommendationMetrics class."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = [0, 1, 2, 3, 4]
        relevant_items = [0, 2, 4, 5, 6]
        
        precision = metrics.precision_at_k(recommendations, relevant_items, k=5)
        assert precision == 0.6  # 3 out of 5 recommendations are relevant
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = [0, 1, 2, 3, 4]
        relevant_items = [0, 2, 4, 5, 6]
        
        recall = metrics.recall_at_k(recommendations, relevant_items, k=5)
        assert recall == 0.6  # 3 out of 5 relevant items are recommended
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = [0, 1, 2, 3, 4]
        relevant_items = [0, 2, 4, 5, 6]
        
        ndcg = metrics.ndcg_at_k(recommendations, relevant_items, k=5)
        assert 0.0 <= ndcg <= 1.0
    
    def test_hit_rate_at_k(self):
        """Test hit rate@k calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = [0, 1, 2, 3, 4]
        relevant_items = [0, 2, 4, 5, 6]
        
        hit_rate = metrics.hit_rate_at_k(recommendations, relevant_items, k=5)
        assert hit_rate == 1.0  # At least one relevant item is recommended
        
        # Test case with no hits
        recommendations = [1, 3, 5, 7, 9]
        hit_rate = metrics.hit_rate_at_k(recommendations, relevant_items, k=5)
        assert hit_rate == 0.0  # No relevant items are recommended
    
    def test_coverage(self):
        """Test coverage calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = {
            0: [0, 1, 2],
            1: [1, 2, 3],
            2: [2, 3, 4]
        }
        all_items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        coverage = metrics.coverage(recommendations, all_items)
        assert coverage == 0.5  # 5 out of 10 items are recommended


class TestSetSeed:
    """Test set_seed function."""
    
    def test_set_seed(self):
        """Test seed setting."""
        # This is a basic test - in practice, you'd want to test actual randomness
        set_seed(42)
        # If no exception is raised, the function works
        assert True


# Integration tests
class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        # Generate synthetic data
        generator = DataGenerator(n_users=10, n_items=5, n_interactions=20, random_state=42)
        interactions = generator.generate_interactions()
        
        # Preprocess data
        data_loader = DataLoader(".")
        processed = data_loader.preprocess_interactions(interactions)
        
        # Split data
        train_df, val_df, test_df = data_loader.split_data(processed)
        
        # Train model
        model = PopularityRecommender()
        model.fit(train_df)
        
        # Generate recommendations
        recommendations = model.recommend(0, top_k=3)
        
        assert len(recommendations) <= 3
        assert all(isinstance(item_id, int) for item_id, _ in recommendations)
    
    def test_model_evaluation(self):
        """Test model evaluation pipeline."""
        # Create test data
        interactions = pd.DataFrame({
            'user_id': [0, 0, 1, 1, 2, 2],
            'item_id': [0, 1, 0, 2, 0, 1],
            'rating': [5, 4, 3, 2, 1, 5]
        })
        
        test_interactions = pd.DataFrame({
            'user_id': [0, 1],
            'item_id': [2, 1],
            'rating': [3, 4]
        })
        
        # Train model
        model = PopularityRecommender()
        model.fit(interactions)
        
        # Evaluate model
        from src.utils.metrics import evaluate_model
        metrics = evaluate_model(model, test_interactions, [0, 1, 2])
        
        assert 'precision@5' in metrics
        assert 'recall@5' in metrics
        assert 'ndcg@5' in metrics


if __name__ == "__main__":
    pytest.main([__file__])
