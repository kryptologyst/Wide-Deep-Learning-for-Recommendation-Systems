"""Data utilities for Wide & Deep Learning Recommendation System."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class DataGenerator:
    """Generate synthetic recommendation data for demonstration purposes."""
    
    def __init__(
        self,
        n_users: int = 1000,
        n_items: int = 500,
        n_interactions: int = 10000,
        rating_scale: Tuple[int, int] = (1, 5),
        sparsity: float = 0.95,
        random_state: int = 42
    ) -> None:
        """Initialize data generator.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            n_interactions: Number of interactions to generate
            rating_scale: Min and max rating values
            sparsity: Desired sparsity level
            random_state: Random seed
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_interactions = n_interactions
        self.rating_scale = rating_scale
        self.sparsity = sparsity
        self.random_state = random_state
        
        set_seed(random_state)
    
    def generate_interactions(self) -> pd.DataFrame:
        """Generate synthetic user-item interactions.
        
        Returns:
            DataFrame with columns: user_id, item_id, rating, timestamp
        """
        logger.info(f"Generating {self.n_interactions} interactions...")
        
        # Generate user-item pairs
        user_ids = np.random.randint(0, self.n_users, self.n_interactions)
        item_ids = np.random.randint(0, self.n_items, self.n_interactions)
        
        # Remove duplicates
        pairs = set(zip(user_ids, item_ids))
        while len(pairs) < self.n_interactions:
            new_user = np.random.randint(0, self.n_users)
            new_item = np.random.randint(0, self.n_items)
            pairs.add((new_user, new_item))
        
        pairs = list(pairs)[:self.n_interactions]
        user_ids, item_ids = zip(*pairs)
        
        # Generate ratings with some patterns
        ratings = self._generate_ratings(user_ids, item_ids)
        
        # Generate timestamps
        timestamps = np.random.randint(
            1609459200,  # 2021-01-01
            1672531200,  # 2023-01-01
            self.n_interactions
        )
        
        df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings,
            'timestamp': timestamps
        })
        
        logger.info(f"Generated interactions with sparsity: {1 - len(df) / (self.n_users * self.n_items):.3f}")
        return df
    
    def _generate_ratings(self, user_ids: List[int], item_ids: List[int]) -> np.ndarray:
        """Generate ratings with some realistic patterns.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            
        Returns:
            Array of ratings
        """
        ratings = []
        
        for user_id, item_id in zip(user_ids, item_ids):
            # Add some user bias
            user_bias = np.random.normal(0, 0.5)
            
            # Add some item bias
            item_bias = np.random.normal(0, 0.3)
            
            # Add some interaction effect
            interaction_effect = np.random.normal(0, 0.2)
            
            # Base rating
            base_rating = 3.0 + user_bias + item_bias + interaction_effect
            
            # Clip to rating scale and round
            rating = np.clip(base_rating, self.rating_scale[0], self.rating_scale[1])
            rating = round(rating)
            
            ratings.append(rating)
        
        return np.array(ratings)
    
    def generate_items(self) -> pd.DataFrame:
        """Generate synthetic item metadata.
        
        Returns:
            DataFrame with item metadata
        """
        logger.info(f"Generating {self.n_items} items...")
        
        # Generate categories
        categories = ['Electronics', 'Books', 'Clothing', 'Home', 'Sports', 'Beauty', 'Toys', 'Food']
        item_categories = np.random.choice(categories, self.n_items)
        
        # Generate prices
        prices = np.random.lognormal(3, 1, self.n_items)
        
        # Generate titles
        titles = [f"Item_{i:04d}" for i in range(self.n_items)]
        
        df = pd.DataFrame({
            'item_id': range(self.n_items),
            'title': titles,
            'category': item_categories,
            'price': prices,
            'description': [f"Description for {title}" for title in titles]
        })
        
        return df
    
    def generate_users(self) -> pd.DataFrame:
        """Generate synthetic user metadata.
        
        Returns:
            DataFrame with user metadata
        """
        logger.info(f"Generating {self.n_users} users...")
        
        # Generate demographics
        ages = np.random.randint(18, 80, self.n_users)
        genders = np.random.choice(['M', 'F', 'Other'], self.n_users)
        
        # Generate locations
        locations = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'CN']
        user_locations = np.random.choice(locations, self.n_users)
        
        df = pd.DataFrame({
            'user_id': range(self.n_users),
            'age': ages,
            'gender': genders,
            'location': user_locations,
            'registration_date': np.random.randint(
                1609459200,  # 2021-01-01
                1672531200,  # 2023-01-01
                self.n_users
            )
        })
        
        return df


class DataLoader:
    """Load and preprocess recommendation data."""
    
    def __init__(self, data_dir: Union[str, Path]) -> None:
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
    
    def load_interactions(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load interactions data.
        
        Args:
            file_path: Path to interactions file
            
        Returns:
            DataFrame with interactions
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Interactions file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        required_cols = ['user_id', 'item_id', 'rating']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        logger.info(f"Loaded {len(df)} interactions")
        return df
    
    def load_items(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load items data.
        
        Args:
            file_path: Path to items file
            
        Returns:
            DataFrame with items metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Items file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} items")
        return df
    
    def load_users(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load users data.
        
        Args:
            file_path: Path to users file
            
        Returns:
            DataFrame with users metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Users file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} users")
        return df
    
    def preprocess_interactions(
        self,
        df: pd.DataFrame,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5
    ) -> pd.DataFrame:
        """Preprocess interactions data.
        
        Args:
            df: Interactions DataFrame
            min_user_interactions: Minimum interactions per user
            min_item_interactions: Minimum interactions per item
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing interactions...")
        
        # Filter users with minimum interactions
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df['user_id'].isin(valid_users)]
        
        # Filter items with minimum interactions
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df['item_id'].isin(valid_items)]
        
        # Encode user and item IDs
        df['user_id'] = self.user_encoder.fit_transform(df['user_id'])
        df['item_id'] = self.item_encoder.fit_transform(df['item_id'])
        
        logger.info(f"After preprocessing: {len(df)} interactions, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
        return df
    
    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets.
        
        Args:
            df: Interactions DataFrame
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data...")
        
        # First split: train+val vs test
        try:
            train_val_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state, stratify=df['user_id']
            )
        except ValueError:
            # If stratification fails (e.g., some users have only 1 interaction), use random split
            train_val_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state
            )
        
        # Second split: train vs val
        try:
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_size/(1-test_size), random_state=random_state, stratify=train_val_df['user_id']
            )
        except ValueError:
            # If stratification fails, use random split
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_size/(1-test_size), random_state=random_state
            )
        
        logger.info(f"Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        return train_df, val_df, test_df


def save_data(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """Save DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.info(f"Saved data to {file_path}")


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load DataFrame from CSV.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded DataFrame
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data from {file_path}: {len(df)} rows")
    return df
