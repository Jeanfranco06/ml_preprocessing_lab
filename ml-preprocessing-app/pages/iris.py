"""
Iris Dataset Processing Page
This module contains specific functions for Iris dataset preprocessing and visualization.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from utils.preprocessing import get_dataset_config
from utils.visualization import show_initial_exploration, show_data_cleaning

def load_iris_data():
    """Load Iris dataset from sklearn"""
    iris = load_iris()
    df = pd.DataFrame(
        data=np.c_[iris['data'], iris['target']],
        columns=iris['feature_names'] + ['target']
    )
    # Add species names for better readability
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['target'].map(species_map)
    return df

def preprocess_iris(df):
    """Complete preprocessing pipeline for Iris dataset"""
    config = get_dataset_config("iris")

    # For Iris, minimal preprocessing is needed as it's already clean
    # Just ensure no missing values (though there shouldn't be any)
    df_clean = df.copy()

    return df_clean

def show_iris_insights(df):
    """Show Iris-specific insights and visualizations"""
    st.header("🌸 Iris Species Classification Analysis")

    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Species Count", df['species'].nunique())
    with col3:
        st.metric("Features", len(df.columns) - 2)  # Excluding target and species
    with col4:
        st.metric("Avg Sepal Length", f"{df['sepal length (cm)'].mean():.2f}cm")

    # Species distribution
    st.subheader("Species Distribution")
    species_counts = df['species'].value_counts()
    st.bar_chart(species_counts)

    # Feature distributions by species
    st.subheader("Feature Distributions by Species")

    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    selected_feature = st.selectbox("Select feature to analyze", features)

    fig, ax = plt.subplots(figsize=(10, 6))
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        ax.hist(subset[selected_feature], alpha=0.7, label=species, bins=15)
    ax.legend()
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    ax.set_title(f"{selected_feature} Distribution by Species")
    st.pyplot(fig)

    # Scatter plots
    st.subheader("Feature Relationships")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Sepal Length vs Sepal Width")
        fig, ax = plt.subplots()
        for species in df['species'].unique():
            subset = df[df['species'] == species]
            ax.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'],
                      label=species, alpha=0.7)
        ax.legend()
        ax.set_xlabel("Sepal Length (cm)")
        ax.set_ylabel("Sepal Width (cm)")
        ax.set_title("Sepal Dimensions by Species")
        st.pyplot(fig)

    with col2:
        st.write("Petal Length vs Petal Width")
        fig, ax = plt.subplots()
        for species in df['species'].unique():
            subset = df[df['species'] == species]
            ax.scatter(subset['petal length (cm)'], subset['petal width (cm)'],
                      label=species, alpha=0.7)
        ax.legend()
        ax.set_xlabel("Petal Length (cm)")
        ax.set_ylabel("Petal Width (cm)")
        ax.set_title("Petal Dimensions by Species")
        st.pyplot(fig)

    # Correlation matrix
    st.subheader("Feature Correlations")
    numerical_features = [col for col in df.columns if col not in ['target', 'species']]
    corr_matrix = df[numerical_features].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
    ax.set_title("Correlation Matrix of Iris Features")
    st.pyplot(fig)

    # Pair plot (if user wants to see all relationships)
    if st.checkbox("Show Pair Plot (All Feature Relationships)"):
        st.write("This may take a moment to render...")
        fig = sns.pairplot(df, hue='species', diag_kind='kde')
        st.pyplot(fig)

def main():
    """Main function for Iris page"""
    st.title("🌸 Iris Dataset Analysis")

    # Load data
    df = load_iris_data()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Exploration", "🧹 Preprocessing", "📈 Insights"])

    with tab1:
        show_initial_exploration(df, "iris")

    with tab2:
        cleaned_df = show_data_cleaning(df, "iris")
        if cleaned_df is not None:
            st.subheader("Processed Data Preview")
            st.dataframe(cleaned_df.head())

    with tab3:
        if cleaned_df is not None:
            show_iris_insights(cleaned_df)
        else:
            show_iris_insights(df)

if __name__ == "__main__":
    main()
