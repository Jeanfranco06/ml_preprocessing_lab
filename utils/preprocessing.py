import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import streamlit as st

def get_dataset_config(dataset_key):
    """Get configuration for each dataset"""
    configs = {
        "titanic": {
            "irrelevant_cols": ["Name", "Ticket", "Cabin", "PassengerId"],
            "target_col": "Survived",
            "categorical_cols": ["Sex", "Embarked"],
            "numerical_cols": ["Age", "Fare", "SibSp", "Parch"],
            "encoding_method": "label"  # label for binary, onehot for multi
        },
        "student_performance": {
            "irrelevant_cols": [],
            "target_col": "G3",
            "categorical_cols": ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
                               "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
                               "nursery", "higher", "internet", "romantic"],
            "numerical_cols": ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                             "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2"],
            "encoding_method": "onehot"
        },
        "iris": {
            "irrelevant_cols": [],
            "target_col": "target",
            "categorical_cols": [],
            "numerical_cols": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
            "encoding_method": "none"
        }
    }
    return configs.get(dataset_key, {})

def handle_missing_values(df, config):
    """Handle missing values based on dataset configuration"""
    df_clean = df.copy()

    # For numerical columns, fill with mean
    for col in config["numerical_cols"]:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            mean_val = df_clean[col].mean()
            df_clean[col].fillna(mean_val, inplace=True)

    # For categorical columns, fill with mode
    for col in config["categorical_cols"]:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)

    return df_clean

def remove_duplicates(df):
    """Remove duplicate rows"""
    initial_shape = df.shape
    df_clean = df.drop_duplicates()
    final_shape = df_clean.shape
    duplicates_removed = initial_shape[0] - final_shape[0]
    return df_clean, duplicates_removed

def detect_outliers(df, numerical_cols, method="iqr"):
    """Detect outliers using IQR method"""
    outliers_count = {}
    df_clean = df.copy()

    for col in numerical_cols:
        if col in df_clean.columns:
            if method == "iqr":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
                outliers_count[col] = len(outliers)

    return outliers_count

def encode_categorical_variables(df, config):
    """Encode categorical variables"""
    df_encoded = df.copy()

    if config["encoding_method"] == "label":
        le = LabelEncoder()
        for col in config["categorical_cols"]:
            if col in df_encoded.columns:
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    elif config["encoding_method"] == "onehot":
        # Get categorical columns that exist in the dataframe
        existing_cat_cols = [col for col in config["categorical_cols"] if col in df_encoded.columns]

        if existing_cat_cols:
            # Create dummy variables
            dummies = pd.get_dummies(df_encoded[existing_cat_cols], prefix=existing_cat_cols, drop_first=False)
            df_encoded = pd.concat([df_encoded.drop(existing_cat_cols, axis=1), dummies], axis=1)

    return df_encoded

def normalize_data(df, numerical_cols, method="standard"):
    """Normalize numerical data"""
    df_normalized = df.copy()

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        return df_normalized

    # Only scale numerical columns that exist
    existing_num_cols = [col for col in numerical_cols if col in df_normalized.columns]

    if existing_num_cols:
        df_normalized[existing_num_cols] = scaler.fit_transform(df_normalized[existing_num_cols])

    return df_normalized, scaler

def split_data(df, target_col, test_size=0.3, random_state=42):
    """Split data into train and test sets"""
    if target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() < 10 else None
        )

        return X_train, X_test, y_train, y_test
    else:
        # If no target column, split the entire dataset
        X_train, X_test = train_test_split(df, test_size=test_size, random_state=random_state)
        return X_train, X_test, None, None
