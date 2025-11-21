"""Streamlit demo for Wide & Deep Learning Recommendation System."""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.utils import DataLoader as DataLoaderUtil, load_data
from src.models.baselines import create_baseline_model
from src.models.wide_deep import WideAndDeepTrainer, WideAndDeepModel
from src.utils.metrics import RecommendationMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_data
def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@st.cache_data
def load_data_files(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data files."""
    data_config = config['data']
    
    interactions = load_data(data_config['interactions_file'])
    items = load_data(data_config['items_file'])
    users = load_data(data_config['users_file'])
    
    return interactions, items, users


@st.cache_resource
def load_models(config: Dict, interactions: pd.DataFrame) -> Dict:
    """Load trained models."""
    models = {}
    
    # Load baseline models
    baseline_config = config['model']['baselines']
    
    for model_name, model_config in baseline_config.items():
        if not model_config.get('enabled', False):
            continue
        
        try:
            model = create_baseline_model(model_name, **{k: v for k, v in model_config.items() if k != 'enabled'})
            model.fit(interactions)
            models[model_name] = model
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
    
    # Load Wide & Deep model
    try:
        model_path = Path("models/checkpoints/wide_deep_model.pth")
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            model_config = checkpoint['model_config']
            
            wide_deep_model = WideAndDeepModel(**model_config)
            trainer = WideAndDeepTrainer(wide_deep_model, device='cpu')
            trainer.load_model(str(model_path))
            
            models['wide_deep'] = trainer
        else:
            logger.warning("Wide & Deep model not found")
    except Exception as e:
        logger.warning(f"Failed to load Wide & Deep model: {e}")
    
    return models


def create_user_item_interaction_plot(interactions: pd.DataFrame, items: pd.DataFrame) -> go.Figure:
    """Create user-item interaction heatmap."""
    # Sample users and items for visualization
    sample_users = interactions['user_id'].unique()[:20]
    sample_items = interactions['item_id'].unique()[:20]
    
    sample_interactions = interactions[
        (interactions['user_id'].isin(sample_users)) & 
        (interactions['item_id'].isin(sample_items))
    ]
    
    # Create pivot table
    pivot_table = sample_interactions.pivot_table(
        index='user_id', 
        columns='item_id', 
        values='rating', 
        fill_value=0
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=[f"Item {i}" for i in pivot_table.columns],
        y=[f"User {i}" for i in pivot_table.index],
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title="User-Item Interaction Matrix (Sample)",
        xaxis_title="Items",
        yaxis_title="Users",
        height=500
    )
    
    return fig


def create_item_popularity_plot(interactions: pd.DataFrame, items: pd.DataFrame) -> go.Figure:
    """Create item popularity plot."""
    item_stats = interactions.groupby('item_id').agg({
        'rating': ['mean', 'count']
    }).round(2)
    
    item_stats.columns = ['avg_rating', 'interaction_count']
    item_stats = item_stats.reset_index()
    
    # Merge with item metadata
    item_stats = item_stats.merge(items, on='item_id', how='left')
    
    fig = px.scatter(
        item_stats,
        x='interaction_count',
        y='avg_rating',
        color='category',
        size='interaction_count',
        hover_data=['title'],
        title="Item Popularity vs Average Rating"
    )
    
    fig.update_layout(
        xaxis_title="Number of Interactions",
        yaxis_title="Average Rating",
        height=500
    )
    
    return fig


def create_model_comparison_plot(results_df: pd.DataFrame) -> go.Figure:
    """Create model comparison plot."""
    if results_df.empty:
        return go.Figure()
    
    # Select key metrics for comparison
    key_metrics = ['precision@10', 'recall@10', 'ndcg@10', 'hit_rate@10']
    available_metrics = [m for m in key_metrics if m in results_df.columns]
    
    if not available_metrics:
        return go.Figure()
    
    fig = go.Figure()
    
    for metric in available_metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=results_df.index,
            y=results_df[metric],
            text=results_df[metric].round(3),
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        barmode='group',
        height=500
    )
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Wide & Deep Recommendation System",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Wide & Deep Learning for Recommendations")
    st.markdown("Interactive demo of Wide & Deep Learning recommendation system")
    
    # Load configuration and data
    try:
        config = load_config("configs/config.yaml")
        interactions, items, users = load_data_files(config)
        
        st.success("✅ Data loaded successfully!")
        
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.stop()
    
    # Sidebar
    st.sidebar.header("📊 Data Overview")
    st.sidebar.metric("Users", len(users))
    st.sidebar.metric("Items", len(items))
    st.sidebar.metric("Interactions", len(interactions))
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Data Analysis", "🎯 Recommendations", "📊 Model Comparison", "🔍 Item Search"])
    
    with tab1:
        st.header("📈 Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("User-Item Interactions")
            interaction_plot = create_user_item_interaction_plot(interactions, items)
            st.plotly_chart(interaction_plot, use_container_width=True)
        
        with col2:
            st.subheader("Item Popularity")
            popularity_plot = create_item_popularity_plot(interactions, items)
            st.plotly_chart(popularity_plot, use_container_width=True)
        
        # Data statistics
        st.subheader("📊 Data Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Rating", f"{interactions['rating'].mean():.2f}")
            st.metric("Rating Std", f"{interactions['rating'].std():.2f}")
        
        with col2:
            st.metric("Sparsity", f"{1 - len(interactions) / (len(users) * len(items)):.3f}")
            st.metric("Interactions per User", f"{len(interactions) / len(users):.1f}")
        
        with col3:
            st.metric("Interactions per Item", f"{len(interactions) / len(items):.1f}")
            st.metric("Unique Categories", items['category'].nunique())
    
    with tab2:
        st.header("🎯 Generate Recommendations")
        
        # Load models
        with st.spinner("Loading models..."):
            models = load_models(config, interactions)
        
        if not models:
            st.error("❌ No models available. Please train models first.")
            st.stop()
        
        # User selection
        st.subheader("Select User")
        user_options = interactions['user_id'].unique()
        selected_user = st.selectbox("Choose a user:", user_options)
        
        # Model selection
        st.subheader("Select Model")
        model_options = list(models.keys())
        selected_model = st.selectbox("Choose a model:", model_options)
        
        # Recommendation parameters
        col1, col2 = st.columns(2)
        
        with col1:
            top_k = st.slider("Number of recommendations:", 1, 20, 10)
        
        with col2:
            show_explanations = st.checkbox("Show explanations", value=True)
        
        # Generate recommendations
        if st.button("🎯 Generate Recommendations"):
            try:
                model = models[selected_model]
                
                # Get user's interaction history
                user_interactions = interactions[interactions['user_id'] == selected_user]
                user_items = user_interactions['item_id'].tolist()
                
                # Generate recommendations
                recommendations = model.recommend(selected_user, top_k=top_k)
                
                st.subheader(f"📋 Recommendations for User {selected_user}")
                
                # Display recommendations
                for i, (item_id, score) in enumerate(recommendations, 1):
                    item_info = items[items['item_id'] == item_id].iloc[0]
                    
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
                        st.write(f"**#{i}**")
                    
                    with col2:
                        st.write(f"**{item_info['title']}**")
                        st.write(f"Category: {item_info['category']}")
                        st.write(f"Price: ${item_info['price']:.2f}")
                        
                        if show_explanations:
                            st.write(f"*Score: {score:.3f}*")
                    
                    with col3:
                        if item_id in user_items:
                            st.write("✅ **Rated**")
                        else:
                            st.write("🆕 **New**")
                
                # User's interaction history
                st.subheader("📚 User's Interaction History")
                
                if len(user_interactions) > 0:
                    user_items_df = user_interactions.merge(items, on='item_id')
                    
                    for _, row in user_items_df.iterrows():
                        col1, col2, col3 = st.columns([1, 3, 1])
                        
                        with col1:
                            st.write(f"⭐ {row['rating']}")
                        
                        with col2:
                            st.write(f"**{row['title']}**")
                            st.write(f"Category: {row['category']}")
                        
                        with col3:
                            st.write(f"${row['price']:.2f}")
                else:
                    st.write("No interaction history available.")
                
            except Exception as e:
                st.error(f"❌ Failed to generate recommendations: {e}")
    
    with tab3:
        st.header("📊 Model Performance Comparison")
        
        # Load results
        try:
            results_path = Path("models/logs/results.csv")
            if results_path.exists():
                results_df = pd.read_csv(results_path, index_col=0)
                
                st.subheader("📈 Performance Metrics")
                
                # Create comparison plot
                comparison_plot = create_model_comparison_plot(results_df)
                if comparison_plot.data:
                    st.plotly_chart(comparison_plot, use_container_width=True)
                
                # Display results table
                st.subheader("📋 Detailed Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv()
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="model_results.csv",
                    mime="text/csv"
                )
                
            else:
                st.warning("⚠️ No results file found. Please train models first.")
                
        except Exception as e:
            st.error(f"❌ Failed to load results: {e}")
    
    with tab4:
        st.header("🔍 Item Search & Similarity")
        
        # Item search
        st.subheader("Search Items")
        
        search_term = st.text_input("Enter search term:")
        
        if search_term:
            # Simple text search
            matching_items = items[
                items['title'].str.contains(search_term, case=False) |
                items['description'].str.contains(search_term, case=False)
            ]
            
            if len(matching_items) > 0:
                st.write(f"Found {len(matching_items)} items:")
                
                for _, item in matching_items.iterrows():
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
                        st.write(f"**{item['item_id']}**")
                    
                    with col2:
                        st.write(f"**{item['title']}**")
                        st.write(f"Category: {item['category']}")
                        st.write(f"Description: {item['description']}")
                    
                    with col3:
                        st.write(f"${item['price']:.2f}")
                        
                        if st.button(f"Find Similar", key=f"similar_{item['item_id']}"):
                            # Simple similarity based on category
                            similar_items = items[
                                (items['category'] == item['category']) & 
                                (items['item_id'] != item['item_id'])
                            ].head(5)
                            
                            st.write("**Similar Items:**")
                            for _, similar in similar_items.iterrows():
                                st.write(f"- {similar['title']} (${similar['price']:.2f})")
            
            else:
                st.write("No items found matching your search.")
        
        # Category analysis
        st.subheader("📊 Category Analysis")
        
        category_stats = items.groupby('category').agg({
            'item_id': 'count',
            'price': ['mean', 'std']
        }).round(2)
        
        category_stats.columns = ['count', 'avg_price', 'price_std']
        category_stats = category_stats.reset_index()
        
        fig = px.bar(
            category_stats,
            x='category',
            y='count',
            title="Items per Category"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Price distribution
        fig = px.box(
            items,
            x='category',
            y='price',
            title="Price Distribution by Category"
        )
        
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
