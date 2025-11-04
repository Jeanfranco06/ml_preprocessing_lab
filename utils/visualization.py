# Módulo de visualización básica
# Funciones simples para gráficos básicos

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def create_basic_plot(df, column):
    """Crear un gráfico básico para una columna"""
    if column in df.columns and df[column].dtype in ['int64', 'float64']:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[column], bins=20, alpha=0.7, color='#3498db', edgecolor='black')
        ax.set_title(f'Distribución de {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frecuencia')
        ax.grid(True, alpha=0.3)
        return fig
    return None
