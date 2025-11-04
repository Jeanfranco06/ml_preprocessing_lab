import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from .preprocessing import get_dataset_config

def show_initial_exploration(df, dataset_key):
    """Display initial data exploration"""
    config = get_dataset_config(dataset_key)

    # Basic info
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        null_count = df.isnull().sum().sum()
        st.metric("Total Nulls", null_count)
    with col4:
        dup_count = df.duplicated().sum()
        st.metric("Duplicates", dup_count)

    # Data types
    st.subheader("Data Types")
    dtypes_df = pd.DataFrame({
        'Column': df.dtypes.index,
        'Type': df.dtypes.values,
        'Non-Null Count': df.notnull().sum().values,
        'Null Count': df.isnull().sum().values
    })
    st.dataframe(dtypes_df)

    # Sample data
    st.subheader("Sample Data (First 5 rows)")
    st.dataframe(df.head())

    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())

    # Missing values heatmap
    st.subheader("Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
    st.pyplot(fig)

    # Distribution of numerical variables
    numerical_cols = config.get("numerical_cols", [])
    existing_num_cols = [col for col in numerical_cols if col in df.columns]

    if existing_num_cols:
        st.subheader("Distribution of Numerical Variables")
        selected_col = st.selectbox("Select column to visualize", existing_num_cols)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        sns.histplot(df[selected_col], kde=True, ax=ax1)
        ax1.set_title(f'Histogram of {selected_col}')

        # Box plot
        sns.boxplot(y=df[selected_col], ax=ax2)
        ax2.set_title(f'Box Plot of {selected_col}')

        st.pyplot(fig)

def show_data_cleaning(df, dataset_key):
    """Display data cleaning process"""
    config = get_dataset_config(dataset_key)

    # Remove irrelevant columns
    df_clean = df.copy()
    if config.get("irrelevant_cols"):
        cols_to_remove = [col for col in config["irrelevant_cols"] if col in df_clean.columns]
        if cols_to_remove:
            st.subheader("Removing Irrelevant Columns")
            st.write(f"Columns to remove: {cols_to_remove}")
            df_clean = df_clean.drop(cols_to_remove, axis=1)

    # Handle duplicates
    st.subheader("Removing Duplicates")
    df_clean, duplicates_removed = remove_duplicates(df_clean)
    if duplicates_removed > 0:
        st.success(f"Removed {duplicates_removed} duplicate rows")
    else:
        st.info("No duplicate rows found")

    # Handle missing values
    st.subheader("Handling Missing Values")
    null_before = df_clean.isnull().sum().sum()

    if null_before > 0:
        df_clean = handle_missing_values(df_clean, config)
        null_after = df_clean.isnull().sum().sum()
        st.success(f"Handled {null_before - null_after} missing values")
    else:
        st.info("No missing values found")

    # Show before/after comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before Cleaning")
        st.metric("Shape", f"{df.shape[0]} x {df.shape[1]}")
        st.metric("Null Values", df.isnull().sum().sum())

    with col2:
        st.subheader("After Cleaning")
        st.metric("Shape", f"{df_clean.shape[0]} x {df_clean.shape[1]}")
        st.metric("Null Values", df_clean.isnull().sum().sum())

    return df_clean

def show_encoding(df, dataset_key):
    """Display encoding process"""
    config = get_dataset_config(dataset_key)

    if not config.get("categorical_cols"):
        st.info("No categorical columns to encode for this dataset")
        return df

    st.subheader("Categorical Variable Encoding")

    # Show original categorical columns
    existing_cat_cols = [col for col in config["categorical_cols"] if col in df.columns]

    if existing_cat_cols:
        st.write("Categorical columns found:", existing_cat_cols)

        # Show unique values for each categorical column
        for col in existing_cat_cols:
            st.write(f"**{col}**: {df[col].unique()}")

        # Apply encoding
        df_encoded = encode_categorical_variables(df, config)

        st.subheader("Encoding Applied")
        if config["encoding_method"] == "label":
            st.write("Applied Label Encoding to categorical variables")
        elif config["encoding_method"] == "onehot":
            st.write("Applied One-Hot Encoding to categorical variables")

        # Show sample of encoded data
        st.subheader("Sample of Encoded Data")
        st.dataframe(df_encoded.head())

        return df_encoded
    else:
        st.info("No categorical columns found in the dataset")
        return df

def show_normalization(df, dataset_key):
    """Display normalization process"""
    config = get_dataset_config(dataset_key)
    numerical_cols = config.get("numerical_cols", [])
    existing_num_cols = [col for col in numerical_cols if col in df.columns]

    if not existing_num_cols:
        st.info("No numerical columns to normalize")
        return df

    st.subheader("Data Normalization/Standardization")

    # Method selection
    method = st.selectbox(
        "Select normalization method",
        ["standard", "minmax"],
        format_func=lambda x: "Standard Scaler (Z-score)" if x == "standard" else "Min-Max Scaler"
    )

    # Apply normalization
    df_normalized, scaler = normalize_data(df, existing_num_cols, method)

    # Show before/after comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Before Normalization")
        st.dataframe(df[existing_num_cols].describe())

    with col2:
        st.subheader("After Normalization")
        st.dataframe(df_normalized[existing_num_cols].describe())

    # Visualization comparison
    st.subheader("Distribution Comparison")
    selected_col = st.selectbox("Select column to compare", existing_num_cols, key="norm_compare")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(df[selected_col], kde=True, ax=ax1, color='blue', alpha=0.7)
    ax1.set_title(f'Original {selected_col}')

    sns.histplot(df_normalized[selected_col], kde=True, ax=ax2, color='red', alpha=0.7)
    ax2.set_title(f'Normalized {selected_col}')

    st.pyplot(fig)

    return df_normalized

def show_data_split(df, dataset_key):
    """Display train/test split"""
    config = get_dataset_config(dataset_key)
    target_col = config.get("target_col")

    # Split configuration
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size (%)", 10, 50, 30) / 100
    with col2:
        random_state = st.number_input("Random State", value=42, min_value=0)

    # Perform split
    X_train, X_test, y_train, y_test = split_data(df, target_col, test_size, random_state)

    # Display results
    st.subheader("Train/Test Split Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("X_train Shape", f"{X_train.shape[0]} x {X_train.shape[1]}")
    with col2:
        st.metric("X_test Shape", f"{X_test.shape[0]} x {X_test.shape[1]}")
    with col3:
        if y_train is not None:
            st.metric("y_train Shape", f"{y_train.shape[0]}")
        else:
            st.metric("y_train", "N/A")
    with col4:
        if y_test is not None:
            st.metric("y_test Shape", f"{y_test.shape[0]}")
        else:
            st.metric("y_test", "N/A")

    # Class distribution (if classification)
    if y_train is not None and y_train.nunique() < 10:
        st.subheader("Class Distribution")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Training Set")
            train_dist = y_train.value_counts().sort_index()
            st.bar_chart(train_dist)

        with col2:
            st.write("Test Set")
            test_dist = y_test.value_counts().sort_index()
            st.bar_chart(test_dist)

def show_advanced_visualizations(df, dataset_key):
    """Display advanced visualizations and insights"""
    config = get_dataset_config(dataset_key)

    st.subheader("Advanced Visualizations & Insights")

    if dataset_key == "titanic":
        # Survival analysis
        if "Survived" in df.columns:
            fig = px.histogram(df, x="Age", color="Survived",
                             title="Age Distribution by Survival",
                             barmode="overlay", opacity=0.7)
            st.plotly_chart(fig)

            # Correlation heatmap
            numerical_cols = [col for col in config["numerical_cols"] if col in df.columns]
            if numerical_cols:
                corr_matrix = df[numerical_cols + ["Survived"]].corr()
                fig = px.imshow(corr_matrix, text_auto=True,
                              title="Correlation Matrix")
                st.plotly_chart(fig)

    elif dataset_key == "student_performance":
        # Grade correlations
        if all(col in df.columns for col in ["G1", "G2", "G3"]):
            fig = px.scatter_3d(df, x="G1", y="G2", z="G3",
                              title="G1 vs G2 vs G3 Relationship")
            st.plotly_chart(fig)

            # Correlation between grades
            grade_corr = df[["G1", "G2", "G3"]].corr()
            st.write("Grade Correlations:")
            st.dataframe(grade_corr)

    elif dataset_key == "iris":
        # Iris scatter plots
        fig = px.scatter(df, x="sepal length (cm)", y="sepal width (cm)",
                        color="target", title="Sepal Length vs Width by Species")
        st.plotly_chart(fig)

        fig2 = px.scatter(df, x="petal length (cm)", y="petal width (cm)",
                         color="target", title="Petal Length vs Width by Species")
        st.plotly_chart(fig2)

# Import functions that are used
from .preprocessing import remove_duplicates, handle_missing_values, encode_categorical_variables, normalize_data, split_data
