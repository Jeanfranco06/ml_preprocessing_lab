"""
Data Preprocessing Module
Funciones para limpieza, transformación y preparación de datos.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def get_dataset_config(dataset_key):
    """
    Obtiene la configuración específica para cada dataset.

    Args:
        dataset_key (str): Identificador del dataset

    Returns:
        dict: Configuración del dataset
    """
    configs = {
        "titanic": {
            "irrelevant_cols": ["Name", "Ticket", "Cabin", "PassengerId"],
            "target_col": "Survived",
            "categorical_cols": ["Sex", "Embarked"],
            "numerical_cols": ["Age", "Fare", "SibSp", "Parch"],
            "test_size": 0.3
        },
        "student_performance": {
            "irrelevant_cols": [],
            "target_col": "G3",
            "categorical_cols": ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
                               "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
                               "nursery", "higher", "internet", "romantic"],
            "numerical_cols": ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                             "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2"],
            "test_size": 0.2
        },
        "iris": {
            "irrelevant_cols": [],
            "target_col": "target",
            "categorical_cols": [],
            "numerical_cols": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
            "test_size": 0.3
        }
    }
    return configs.get(dataset_key, {})

def clean_dataset(df, dataset_key):
    """
    Realiza la limpieza completa del dataset.

    Args:
        df (pd.DataFrame): Dataset a limpiar
        dataset_key (str): Identificador del dataset

    Returns:
        pd.DataFrame: Dataset limpio
    """
    config = get_dataset_config(dataset_key)
    df_clean = df.copy()

    # 1. Eliminar columnas irrelevantes
    if config.get("irrelevant_cols"):
        cols_to_remove = [col for col in config["irrelevant_cols"] if col in df_clean.columns]
        if cols_to_remove:
            df_clean = df_clean.drop(cols_to_remove, axis=1)

    # 2. Eliminar duplicados
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    final_rows = df_clean.shape[0]
    duplicates_removed = initial_rows - final_rows

    # 3. Manejar valores nulos
    if df_clean.isnull().sum().sum() > 0:
        # Numéricos: media
        numerical_cols = [col for col in config.get("numerical_cols", []) if col in df_clean.columns]
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

        # Categóricos: moda
        categorical_cols = [col for col in config.get("categorical_cols", []) if col in df_clean.columns]
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    return df_clean

def encode_categorical_variables(df, dataset_key):
    """
    Codifica las variables categóricas.

    Args:
        df (pd.DataFrame): Dataset a codificar
        dataset_key (str): Identificador del dataset

    Returns:
        tuple: (DataFrame codificado, mapeo de codificación)
    """
    config = get_dataset_config(dataset_key)
    df_encoded = df.copy()

    categorical_cols = [col for col in config.get("categorical_cols", []) if col in df_encoded.columns]

    if not categorical_cols:
        return df_encoded, {}

    # Label Encoding
    le = LabelEncoder()
    encoding_map = {}

    for col in categorical_cols:
        original_values = df_encoded[col].unique()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoded_values = df_encoded[col].unique()
        encoding_map[col] = dict(zip(original_values, encoded_values))

    return df_encoded, encoding_map

def normalize_numerical_features(df, dataset_key):
    """
    Normaliza las características numéricas.

    Args:
        df (pd.DataFrame): Dataset a normalizar
        dataset_key (str): Identificador del dataset

    Returns:
        tuple: (DataFrame normalizado, scaler ajustado)
    """
    config = get_dataset_config(dataset_key)
    df_normalized = df.copy()

    target_col = config.get("target_col")
    numerical_cols = [col for col in config.get("numerical_cols", [])
                     if col in df_normalized.columns and col != target_col]

    if not numerical_cols:
        return df_normalized, None

    # Standard Scaler
    scaler = StandardScaler()
    df_normalized[numerical_cols] = scaler.fit_transform(df_normalized[numerical_cols])

    return df_normalized, scaler

def split_train_test(df, dataset_key, random_state=42):
    """
    Divide el dataset en conjuntos de entrenamiento y prueba.

    Args:
        df (pd.DataFrame): Dataset a dividir
        dataset_key (str): Identificador del dataset
        random_state (int): Semilla para reproducibilidad

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    config = get_dataset_config(dataset_key)
    target_col = config.get("target_col")
    test_size = config.get("test_size", 0.3)

    if target_col not in df.columns:
        raise ValueError(f"Columna target '{target_col}' no encontrada en el dataset")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test

def detect_outliers(df, numerical_cols, method="iqr"):
    """
    Detecta outliers usando el método IQR.

    Args:
        df (pd.DataFrame): Dataset a analizar
        numerical_cols (list): Columnas numéricas
        method (str): Método de detección

    Returns:
        dict: Conteo de outliers por columna
    """
    outliers_count = {}

    for col in numerical_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_count[col] = len(outliers)

    return outliers_count

def get_preprocessing_summary(original_df, cleaned_df, encoded_df, normalized_df, dataset_key):
    """
    Genera un resumen completo del preprocesamiento.

    Args:
        original_df, cleaned_df, encoded_df, normalized_df: DataFrames en cada etapa
        dataset_key (str): Identificador del dataset

    Returns:
        dict: Resumen del preprocesamiento
    """
    summary = {
        "original": {
            "rows": original_df.shape[0],
            "columns": original_df.shape[1],
            "nulls": original_df.isnull().sum().sum()
        }
    }

    if cleaned_df is not None:
        summary["cleaned"] = {
            "rows": cleaned_df.shape[0],
            "columns": cleaned_df.shape[1],
            "nulls": cleaned_df.isnull().sum().sum(),
            "duplicates_removed": original_df.shape[0] - cleaned_df.shape[0]
        }

    if encoded_df is not None:
        summary["encoded"] = {
            "rows": encoded_df.shape[0],
            "columns": encoded_df.shape[1],
            "nulls": encoded_df.isnull().sum().sum()
        }

    if normalized_df is not None:
        summary["normalized"] = {
            "rows": normalized_df.shape[0],
            "columns": normalized_df.shape[1],
            "nulls": normalized_df.isnull().sum().sum()
        }

    return summary
