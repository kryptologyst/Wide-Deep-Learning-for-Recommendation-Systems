"""Wide & Deep Learning model implementation for recommendation systems."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class RecommendationDataset(Dataset):
    """PyTorch Dataset for recommendation data."""
    
    def __init__(
        self,
        interactions: np.ndarray,
        ratings: Optional[np.ndarray] = None,
        negative_samples: Optional[np.ndarray] = None
    ) -> None:
        """Initialize dataset.
        
        Args:
            interactions: Array of user-item interactions
            ratings: Optional ratings array
            negative_samples: Optional negative samples
        """
        self.interactions = interactions
        self.ratings = ratings
        self.negative_samples = negative_samples
        
        if self.ratings is None:
            self.ratings = np.ones(len(self.interactions))
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.interactions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item by index.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (user_id, item_id, rating)
        """
        user_id, item_id = self.interactions[idx]
        rating = self.ratings[idx]
        
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(item_id, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float32)
        )


class WideAndDeepModel(nn.Module):
    """Wide & Deep Learning model for recommendations.
    
    Combines wide linear models (memorization) with deep neural networks (generalization).
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        activation: str = "relu"
    ) -> None:
        """Initialize Wide & Deep model.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Embedding dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            activation: Activation function
        """
        super(WideAndDeepModel, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Wide model components (linear)
        self.wide_user_embedding = nn.Embedding(n_users, 1)
        self.wide_item_embedding = nn.Embedding(n_items, 1)
        
        # Deep model components
        self.deep_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.deep_item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Deep network layers
        input_dim = embedding_dim * 2
        layers = []
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU() if activation == "relu" else nn.Tanh(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        self.deep_network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            user_ids: User ID tensor
            item_ids: Item ID tensor
            
        Returns:
            Predicted ratings
        """
        # Wide model: linear combination
        wide_user = self.wide_user_embedding(user_ids)
        wide_item = self.wide_item_embedding(item_ids)
        wide_output = wide_user + wide_item
        
        # Deep model: neural network
        deep_user = self.deep_user_embedding(user_ids)
        deep_item = self.deep_item_embedding(item_ids)
        deep_input = torch.cat([deep_user, deep_item], dim=-1)
        deep_output = self.deep_network(deep_input)
        
        # Combine wide and deep outputs
        output = wide_output + deep_output
        return output.squeeze(-1)


class WideAndDeepTrainer:
    """Trainer for Wide & Deep model."""
    
    def __init__(
        self,
        model: WideAndDeepModel,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = "cpu"
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: Wide & Deep model
            learning_rate: Learning rate
            weight_decay: Weight decay
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        
        logger.info(f"Initialized trainer on device: {device}")
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> float:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (user_ids, item_ids, ratings) in enumerate(dataloader):
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            ratings = ratings.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, ratings)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        return avg_loss
    
    def validate(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> float:
        """Validate model.
        
        Args:
            dataloader: Validation data loader
            epoch: Current epoch
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in dataloader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch}, Validation Loss: {avg_loss:.4f}")
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        min_delta: float = 0.001
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            min_delta: Minimum delta for early stopping
            
        Returns:
            Training history
        """
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader, epoch + 1)
            history["train_loss"].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader, epoch + 1)
                history["val_loss"].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        logger.info("Training completed!")
        return history
    
    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Make predictions.
        
        Args:
            user_ids: User ID tensor
            item_ids: Item ID tensor
            
        Returns:
            Predicted ratings
        """
        self.model.eval()
        with torch.no_grad():
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            predictions = self.model(user_ids, item_ids)
        return predictions.cpu()
    
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
        self.model.eval()
        
        if candidate_items is None:
            candidate_items = list(range(self.model.n_items))
        
        user_tensor = torch.tensor([user_id] * len(candidate_items), dtype=torch.long)
        item_tensor = torch.tensor(candidate_items, dtype=torch.long)
        
        with torch.no_grad():
            scores = self.predict(user_tensor, item_tensor)
        
        # Sort by score and return top-k
        item_scores = list(zip(candidate_items, scores.tolist()))
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        return item_scores[:top_k]
    
    def save_model(self, file_path: str) -> None:
        """Save model checkpoint.
        
        Args:
            file_path: Path to save model
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'n_users': self.model.n_users,
                'n_items': self.model.n_items,
                'embedding_dim': self.model.embedding_dim
            }
        }
        torch.save(checkpoint, file_path)
        logger.info(f"Model saved to {file_path}")
    
    def load_model(self, file_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            file_path: Path to load model from
        """
        checkpoint = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {file_path}")


def create_model(
    n_users: int,
    n_items: int,
    config: Dict[str, Union[int, float, List[int], str]]
) -> WideAndDeepModel:
    """Create Wide & Deep model from configuration.
    
    Args:
        n_users: Number of users
        n_items: Number of items
        config: Model configuration
        
    Returns:
        Wide & Deep model
    """
    return WideAndDeepModel(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=config.get('embedding_dim', 64),
        hidden_dims=config.get('hidden_dims', [128, 64, 32]),
        dropout=config.get('dropout', 0.2),
        activation=config.get('activation', 'relu')
    )
