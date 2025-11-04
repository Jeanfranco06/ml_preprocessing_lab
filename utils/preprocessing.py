# Este módulo contiene funciones básicas de preprocesamiento
# La aplicación principal maneja la lógica directamente para simplicidad

import pandas as pd
import numpy as np

def get_dataset_info(df):
    """Obtener información básica del dataset"""
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum()
    }
