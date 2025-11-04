"""
Student Performance Dataset Processing Page
This module contains specific functions for Student Performance dataset preprocessing and visualization.
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocessing import get_dataset_config, handle_missing_values, remove_duplicates
from utils.visualization import show_initial_exploration, show_data_cleaning

def load_student_data():
    """Load Student Performance dataset"""
    try:
        df = pd.read_csv('datasets/student-mat.csv')
        return df
    except FileNotFoundError:
        st.error("Student Performance dataset not found. Please place 'student-mat.csv' in the datasets folder.")
        return None

def preprocess_student(df):
    """Complete preprocessing pipeline for Student Performance dataset"""
    config = get_dataset_config("student_performance")

    # Remove duplicates
    df_clean, _ = remove_duplicates(df)

    # Handle missing values
    df_clean = handle_missing_values(df_clean, config)

    return df_clean

def show_student_insights(df):
    """Show Student Performance-specific insights and visualizations"""
    st.header("🎓 Student Performance Analysis")

    if "G3" in df.columns:
        # Grade distribution
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Final Grade (G3)", f"{df['G3'].mean():.1f}")
        with col2:
            st.metric("Highest Grade", df['G3'].max())
        with col3:
            st.metric("Lowest Grade", df['G3'].min())

        # Grade correlations
        if all(col in df.columns for col in ["G1", "G2", "G3"]):
            st.subheader("Grade Progression Analysis")
            fig, ax = plt.subplots(figsize=(10, 6))
            grades_corr = df[["G1", "G2", "G3"]].corr()
            sns.heatmap(grades_corr, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title("Correlation between Grades")
            st.pyplot(fig)

            # Scatter plot G1 vs G3
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("G1 vs G3")
                fig, ax = plt.subplots()
                ax.scatter(df["G1"], df["G3"], alpha=0.6)
                ax.set_xlabel("G1 (First Period Grade)")
                ax.set_ylabel("G3 (Final Grade)")
                ax.set_title("First Period vs Final Grade")
                st.pyplot(fig)

            with col2:
                st.subheader("G2 vs G3")
                fig, ax = plt.subplots()
                ax.scatter(df["G2"], df["G3"], alpha=0.6)
                ax.set_xlabel("G2 (Second Period Grade)")
                ax.set_ylabel("G3 (Final Grade)")
                ax.set_title("Second Period vs Final Grade")
                st.pyplot(fig)

        # Study time vs grades
        if "studytime" in df.columns and "G3" in df.columns:
            st.subheader("Study Time vs Final Grade")
            studytime_grades = df.groupby("studytime")["G3"].mean()
            fig, ax = plt.subplots()
            studytime_grades.plot(kind='bar', ax=ax)
            ax.set_xlabel("Study Time (hours)")
            ax.set_ylabel("Average Final Grade")
            ax.set_title("Study Time vs Average Final Grade")
            st.pyplot(fig)

        # Gender differences
        if "sex" in df.columns and "G3" in df.columns:
            st.subheader("Gender Differences in Performance")
            gender_grades = df.groupby("sex")["G3"].agg(['mean', 'std', 'count'])
            st.dataframe(gender_grades)

            fig, ax = plt.subplots()
            df.boxplot(column="G3", by="sex", ax=ax)
            ax.set_title("Final Grade Distribution by Gender")
            ax.set_xlabel("Gender")
            ax.set_ylabel("Final Grade")
            st.pyplot(fig)

def main():
    """Main function for Student Performance page"""
    st.title("🎓 Student Performance Dataset Analysis")

    # Load data
    df = load_student_data()

    if df is not None:
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Exploration", "🧹 Preprocessing", "📈 Insights"])

        with tab1:
            show_initial_exploration(df, "student_performance")

        with tab2:
            cleaned_df = show_data_cleaning(df, "student_performance")
            if cleaned_df is not None:
                st.subheader("Processed Data Preview")
                st.dataframe(cleaned_df.head())

        with tab3:
            if cleaned_df is not None:
                show_student_insights(cleaned_df)
            else:
                show_student_insights(df)

if __name__ == "__main__":
    main()
