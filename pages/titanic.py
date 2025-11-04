"""
Titanic Dataset Processing Page
This module contains specific functions for Titanic dataset preprocessing and visualization.
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from utils.preprocessing import get_dataset_config, handle_missing_values, remove_duplicates
from utils.visualization import show_initial_exploration, show_data_cleaning

def load_titanic_data():
    """Load Titanic dataset"""
    try:
        df = pd.read_csv('datasets/titanic.csv')
        return df
    except FileNotFoundError:
        st.error("Titanic dataset not found. Please place 'titanic.csv' in the datasets folder.")
        return None

def preprocess_titanic(df):
    """Complete preprocessing pipeline for Titanic dataset"""
    config = get_dataset_config("titanic")

    # Remove irrelevant columns
    df_clean = df.drop(config["irrelevant_cols"], axis=1, errors='ignore')

    # Handle missing values
    df_clean = handle_missing_values(df_clean, config)

    # Remove duplicates
    df_clean, _ = remove_duplicates(df_clean)

    return df_clean

def show_titanic_insights(df):
    """Show Titanic-specific insights and visualizations"""
    st.header("🚢 Titanic Survival Analysis")

    if "Survived" in df.columns:
        # Survival rate
        survival_rate = df["Survived"].mean() * 100
        st.metric("Survival Rate", f"{survival_rate:.1f}%")

        # Survival by gender
        if "Sex" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Survival by Gender")
                gender_survival = df.groupby("Sex")["Survived"].mean()
                st.bar_chart(gender_survival)

            with col2:
                st.subheader("Passenger Class Distribution")
                if "Pclass" in df.columns:
                    pclass_dist = df["Pclass"].value_counts().sort_index()
                    st.bar_chart(pclass_dist)

        # Age distribution by survival
        if "Age" in df.columns:
            st.subheader("Age Distribution by Survival Status")
            fig, ax = plt.subplots(figsize=(10, 6))
            for survived in [0, 1]:
                subset = df[df["Survived"] == survived]
                ax.hist(subset["Age"], alpha=0.7, label=f"{'Survived' if survived else 'Did not survive'}", bins=30)
            ax.legend()
            ax.set_xlabel("Age")
            ax.set_ylabel("Count")
            ax.set_title("Age Distribution by Survival")
            st.pyplot(fig)

def main():
    """Main function for Titanic page"""
    st.title("🚢 Titanic Dataset Analysis")

    # Load data
    df = load_titanic_data()

    if df is not None:
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Exploration", "🧹 Preprocessing", "📈 Insights"])

        with tab1:
            show_initial_exploration(df, "titanic")

        with tab2:
            cleaned_df = show_data_cleaning(df, "titanic")
            if cleaned_df is not None:
                st.subheader("Processed Data Preview")
                st.dataframe(cleaned_df.head())

        with tab3:
            if cleaned_df is not None:
                show_titanic_insights(cleaned_df)
            else:
                show_titanic_insights(df)

if __name__ == "__main__":
    main()
