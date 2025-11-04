"""
Data Loaders Module
Funciones para cargar datasets desde diferentes fuentes.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Configuración de rutas
DATA_DIR = Path(__file__).parent.parent.parent / "datasets"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

def load_titanic_data():
    """
    Carga el dataset de Titanic.

    Returns:
        pd.DataFrame: Dataset de Titanic limpio
    """
    file_path = RAW_DATA_DIR / "titanic.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset Titanic no encontrado en {file_path}")

    df = pd.read_csv(file_path)
    return df

def load_student_performance_data():
    """
    Carga el dataset de Student Performance.

    Returns:
        pd.DataFrame: Dataset de rendimiento estudiantil
    """
    file_path = RAW_DATA_DIR / "student-mat.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset Student Performance no encontrado en {file_path}")

    df = pd.read_csv(file_path)
    return df

def load_iris_data():
    """
    Carga el dataset de Iris.

    Returns:
        pd.DataFrame: Dataset de Iris
    """
    file_path = RAW_DATA_DIR / "iris.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset Iris no encontrado en {file_path}")

    df = pd.read_csv(file_path)
    return df

def load_dataset(dataset_name):
    """
    Carga un dataset por nombre.

    Args:
        dataset_name (str): Nombre del dataset ('titanic', 'student_performance', 'iris')

    Returns:
        pd.DataFrame: Dataset cargado

    Raises:
        ValueError: Si el nombre del dataset no es válido
    """
    loaders = {
        "titanic": load_titanic_data,
        "student_performance": load_student_performance_data,
        "iris": load_iris_data
    }

    if dataset_name not in loaders:
        raise ValueError(f"Dataset '{dataset_name}' no válido. Opciones: {list(loaders.keys())}")

    return loaders[dataset_name]()

def get_dataset_info(df):
    """
    Obtiene información básica del dataset.

    Args:
        df (pd.DataFrame): Dataset a analizar

    Returns:
        dict: Información del dataset
    """
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }

def validate_dataset(df, dataset_name):
    """
    Valida que el dataset tenga la estructura esperada.

    Args:
        df (pd.DataFrame): Dataset a validar
        dataset_name (str): Nombre del dataset

    Returns:
        dict: Resultado de la validación
    """
    validations = {
        "titanic": {
            "required_columns": ["PassengerId", "Survived", "Pclass", "Name"],
            "expected_shape": (891, 12)
        },
        "student_performance": {
            "required_columns": ["school", "sex", "age", "G1", "G2", "G3"],
            "expected_shape": (395, 33)
        },
        "iris": {
            "required_columns": ["sepal length (cm)", "target", "species"],
            "expected_shape": (150, 6)
        }
    }

    if dataset_name not in validations:
        return {"valid": False, "message": f"Dataset '{dataset_name}' no tiene validación definida"}

    validation = validations[dataset_name]
    missing_cols = [col for col in validation["required_columns"] if col not in df.columns]

    if missing_cols:
        return {
            "valid": False,
            "message": f"Columnas faltantes: {missing_cols}",
            "missing_columns": missing_cols
        }

    return {
        "valid": True,
        "message": "Dataset válido",
        "shape": df.shape,
        "columns": len(df.columns)
    }
