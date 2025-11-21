"""Main training script for Wide & Deep Learning Recommendation System."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.utils import DataGenerator, DataLoader as DataLoaderUtil, save_data
from src.models.baselines import create_baseline_model
from src.models.wide_deep import WideAndDeepModel, WideAndDeepTrainer, RecommendationDataset
from src.utils.metrics import evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_synthetic_data(config: Dict) -> None:
    """Generate synthetic data if it doesn't exist.
    
    Args:
        config: Configuration dictionary
    """
    data_config = config['data']
    raw_data_dir = Path(data_config['interactions_file']).parent
    
    # Check if data already exists
    interactions_file = Path(data_config['interactions_file'])
    items_file = Path(data_config['items_file'])
    users_file = Path(data_config['users_file'])
    
    if all(f.exists() for f in [interactions_file, items_file, users_file]):
        logger.info("Data files already exist, skipping generation")
        return
    
    logger.info("Generating synthetic data...")
    
    # Generate data
    generator = DataGenerator(**data_config['synthetic'])
    
    interactions = generator.generate_interactions()
    items = generator.generate_items()
    users = generator.generate_users()
    
    # Save data
    save_data(interactions, interactions_file)
    save_data(items, items_file)
    save_data(users, users_file)
    
    logger.info("Synthetic data generation completed")


def train_baseline_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate baseline models.
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        config: Configuration dictionary
        
    Returns:
        Dictionary of model results
    """
    logger.info("Training baseline models...")
    
    baseline_config = config['model']['baselines']
    results = {}
    
    # Get all items for evaluation
    all_items = list(range(train_df['item_id'].nunique()))
    
    # Train each baseline model
    for model_name, model_config in baseline_config.items():
        if not model_config.get('enabled', False):
            continue
        
        logger.info(f"Training {model_name}...")
        
        try:
            # Create model
            model = create_baseline_model(model_name, **{k: v for k, v in model_config.items() if k != 'enabled'})
            
            # Fit model
            model.fit(train_df)
            
            # Evaluate model
            # Extract k values from metric names
            k_values = []
            for metric in config['evaluation']['metrics']:
                if '@' in metric:
                    try:
                        k_val = int(metric.split('@')[1])
                        if k_val not in k_values:
                            k_values.append(k_val)
                    except ValueError:
                        pass
            
            if not k_values:
                k_values = [5, 10, 20]  # Default k values
            
            metrics = evaluate_model(
                model=model,
                test_interactions=test_df,
                all_items=all_items,
                k_values=k_values
            )
            
            results[model_name] = metrics
            logger.info(f"{model_name} evaluation completed")
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            results[model_name] = {}
    
    return results


def train_wide_deep_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict
) -> Dict[str, float]:
    """Train and evaluate Wide & Deep model.
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        config: Configuration dictionary
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Training Wide & Deep model...")
    
    # Get model configuration
    model_config = config['model']['wide_deep']
    training_config = config['training']
    
    # Get dimensions
    n_users = train_df['user_id'].nunique()
    n_items = train_df['item_id'].nunique()
    
    logger.info(f"Training with {n_users} users and {n_items} items")
    
    # Create model
    model = WideAndDeepModel(
        n_users=n_users,
        n_items=n_items,
        **model_config
    )
    
    # Create datasets
    train_dataset = RecommendationDataset(
        interactions=train_df[['user_id', 'item_id']].values,
        ratings=train_df['rating'].values
    )
    
    val_dataset = RecommendationDataset(
        interactions=val_df[['user_id', 'item_id']].values,
        ratings=val_df['rating'].values
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False
    )
    
    # Create trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = WideAndDeepTrainer(
        model=model,
        learning_rate=training_config['learning_rate'],
        device=device
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config['epochs'],
        early_stopping_patience=training_config['early_stopping']['patience'],
        min_delta=training_config['early_stopping']['min_delta']
    )
    
    # Save model
    model_path = Path("models/checkpoints/wide_deep_model.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_path))
    
    # Evaluate model
    all_items = list(range(n_items))
    
    # Extract k values from metric names
    k_values = []
    for metric in config['evaluation']['metrics']:
        if '@' in metric:
            try:
                k_val = int(metric.split('@')[1])
                if k_val not in k_values:
                    k_values.append(k_val)
            except ValueError:
                pass
    
    if not k_values:
        k_values = [5, 10, 20]  # Default k values
    
    metrics = evaluate_model(
        model=trainer,
        test_interactions=test_df,
        all_items=all_items,
        k_values=k_values
    )
    
    logger.info("Wide & Deep model training completed")
    return metrics


def create_results_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create results comparison table.
    
    Args:
        results: Dictionary of model results
        
    Returns:
        Results DataFrame
    """
    if not results:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(results).T
    
    # Round numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(4)
    
    # Sort by NDCG@10 if available
    if 'ndcg@10' in df.columns:
        df = df.sort_values('ndcg@10', ascending=False)
    
    return df


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Wide & Deep Recommendation System")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Configuration file path")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic data")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline model training")
    parser.add_argument("--skip-wide-deep", action="store_true", help="Skip Wide & Deep model training")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    np.random.seed(config['evaluation']['random_state'])
    torch.manual_seed(config['evaluation']['random_state'])
    
    # Generate data if requested or if it doesn't exist
    if args.generate_data:
        generate_synthetic_data(config)
    
    # Load data
    data_config = config['data']
    data_loader = DataLoaderUtil(data_config['interactions_file'])
    
    interactions = data_loader.load_interactions(data_config['interactions_file'])
    items = data_loader.load_items(data_config['items_file'])
    users = data_loader.load_users(data_config['users_file'])
    
    # Preprocess interactions
    interactions = data_loader.preprocess_interactions(interactions)
    
    # Split data
    train_df, val_df, test_df = data_loader.split_data(
        interactions,
        test_size=config['training']['test_split'],
        val_size=config['training']['validation_split'],
        random_state=config['evaluation']['random_state']
    )
    
    logger.info(f"Data split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    # Store results
    all_results = {}
    
    # Train baseline models
    if not args.skip_baselines:
        baseline_results = train_baseline_models(train_df, val_df, test_df, config)
        all_results.update(baseline_results)
    
    # Train Wide & Deep model
    if not args.skip_wide_deep:
        wide_deep_results = train_wide_deep_model(train_df, val_df, test_df, config)
        all_results['wide_deep'] = wide_deep_results
    
    # Create results table
    results_df = create_results_table(all_results)
    
    # Save results
    results_path = Path("models/logs/results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path)
    
    # Print results
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(results_df.to_string())
    print("="*80)
    
    logger.info(f"Results saved to {results_path}")
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
