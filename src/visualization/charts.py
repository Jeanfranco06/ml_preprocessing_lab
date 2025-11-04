"""
Charts Module
Funciones para crear gráficos y visualizaciones.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configurar estilo de matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_dataset_overview_plot(df, title="Resumen del Dataset"):
    """
    Crea un gráfico general del dataset.

    Args:
        df (pd.DataFrame): Dataset a visualizar
        title (str): Título del gráfico

    Returns:
        matplotlib.figure.Figure: Figura creada
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. Tipos de datos
    dtype_counts = df.dtypes.value_counts()
    ax1.bar(range(len(dtype_counts)), dtype_counts.values)
    ax1.set_xticks(range(len(dtype_counts)))
    ax1.set_xticklabels([str(dtype) for dtype in dtype_counts.index], rotation=45)
    ax1.set_title('Distribución de Tipos de Datos')
    ax1.set_ylabel('Cantidad de Columnas')

    # 2. Valores nulos
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        null_counts = null_counts[null_counts > 0]
        ax2.barh(range(len(null_counts)), null_counts.values)
        ax2.set_yticks(range(len(null_counts)))
        ax2.set_yticklabels(null_counts.index)
        ax2.set_title('Columnas con Valores Nulos')
        ax2.set_xlabel('Cantidad de Nulos')
    else:
        ax2.text(0.5, 0.5, 'Sin valores nulos', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Valores Nulos')

    # 3. Distribución de clases (si es clasificación)
    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Primeras 5
        means = [df[col].mean() for col in numeric_cols]
        ax3.bar(range(len(means)), means)
        ax3.set_xticks(range(len(means)))
        ax3.set_xticklabels(numeric_cols, rotation=45)
        ax3.set_title('Media de Variables Numéricas')
        ax3.set_ylabel('Valor Medio')
    else:
        ax3.text(0.5, 0.5, 'No hay variables numéricas', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Variables Numéricas')

    # 4. Matriz de correlación (si hay suficientes variables numéricas)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        corr_matrix = numeric_df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', ax=ax4, cbar=False)
        ax4.set_title('Matriz de Correlación')
    else:
        ax4.text(0.5, 0.5, 'Insuficientes variables\nnuméricas para correlación',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Correlaciones')

    plt.tight_layout()
    return fig

def create_preprocessing_comparison(df_before, df_after, title="Comparación de Preprocesamiento"):
    """
    Crea un gráfico comparativo antes/después del preprocesamiento.

    Args:
        df_before (pd.DataFrame): Dataset original
        df_after (pd.DataFrame): Dataset procesado
        title (str): Título del gráfico

    Returns:
        matplotlib.figure.Figure: Figura creada
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Métricas antes
    before_metrics = {
        'Filas': df_before.shape[0],
        'Columnas': df_before.shape[1],
        'Nulos': df_before.isnull().sum().sum(),
        'Duplicados': df_before.duplicated().sum()
    }

    # Métricas después
    after_metrics = {
        'Filas': df_after.shape[0],
        'Columnas': df_after.shape[1],
        'Nulos': df_after.isnull().sum().sum(),
        'Duplicados': df_after.duplicated().sum()
    }

    metrics = list(before_metrics.keys())
    before_values = list(before_metrics.values())
    after_values = list(after_metrics.values())

    x = np.arange(len(metrics))
    width = 0.35

    ax1.bar(x - width/2, before_values, width, label='Antes', alpha=0.8, color='salmon')
    ax1.bar(x + width/2, after_values, width, label='Después', alpha=0.8, color='skyblue')
    ax1.set_ylabel('Valor')
    ax1.set_title('Comparación de Métricas')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Porcentajes de cambio
    changes = []
    for i, metric in enumerate(metrics):
        if before_values[i] != 0:
            change = ((after_values[i] - before_values[i]) / before_values[i]) * 100
        else:
            change = 0
        changes.append(change)

    colors = ['red' if x < 0 else 'green' for x in changes]
    bars = ax2.bar(metrics, changes, color=colors, alpha=0.7)
    ax2.set_ylabel('Cambio Porcentual (%)')
    ax2.set_title('Porcentaje de Cambio')
    ax2.grid(True, alpha=0.3)

    # Agregar etiquetas de valor
    for bar, change in zip(bars, changes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                '.1f', ha='center', va='bottom' if height >= 0 else 'top')

    plt.tight_layout()
    return fig

def create_normalization_comparison(df_original, df_normalized, numerical_cols, title="Comparación de Normalización"):
    """
    Crea gráficos comparativos de normalización.

    Args:
        df_original (pd.DataFrame): Datos originales
        df_normalized (pd.DataFrame): Datos normalizados
        numerical_cols (list): Columnas numéricas
        title (str): Título del gráfico

    Returns:
        matplotlib.figure.Figure: Figura creada
    """
    if not numerical_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No hay columnas numéricas para comparar',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig

    n_cols = min(len(numerical_cols), 3)  # Máximo 3 columnas para comparación
    selected_cols = numerical_cols[:n_cols]

    fig, axes = plt.subplots(n_cols, 2, figsize=(12, 4*n_cols))
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight='bold')

    for i, col in enumerate(selected_cols):
        # Datos originales
        ax_orig = axes[i][0] if n_cols > 1 else axes[0]
        ax_orig.hist(df_original[col], bins=20, alpha=0.7, color='salmon', edgecolor='black')
        ax_orig.set_title(f'{col} - Original')
        ax_orig.set_xlabel(col)
        ax_orig.set_ylabel('Frecuencia')
        ax_orig.grid(True, alpha=0.3)

        # Estadísticas originales
        orig_stats = df_original[col].describe()
        ax_orig.text(0.02, 0.98, '.2f',
                    transform=ax_orig.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Datos normalizados
        ax_norm = axes[i][1] if n_cols > 1 else axes[1]
        ax_norm.hist(df_normalized[col], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax_norm.set_title(f'{col} - Normalizado')
        ax_norm.set_xlabel(f'{col} (Normalizado)')
        ax_norm.set_ylabel('Frecuencia')
        ax_norm.grid(True, alpha=0.3)

        # Estadísticas normalizadas
        norm_stats = df_normalized[col].describe()
        ax_norm.text(0.02, 0.98, '.2f',
                    transform=ax_norm.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    return fig

def create_correlation_heatmap(df, title="Matriz de Correlación"):
    """
    Crea un heatmap de correlaciones.

    Args:
        df (pd.DataFrame): Dataset para análisis de correlación
        title (str): Título del gráfico

    Returns:
        matplotlib.figure.Figure: Figura creada
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Insuficientes variables numéricas\npara análisis de correlación',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig

    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = numeric_df.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
               square=True, ax=ax, fmt='.2f', cbar_kws={'shrink': 0.8})
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig

def create_outliers_plot(df, numerical_cols, title="Análisis de Outliers"):
    """
    Crea gráficos para análisis de outliers.

    Args:
        df (pd.DataFrame): Dataset a analizar
        numerical_cols (list): Columnas numéricas
        title (str): Título del gráfico

    Returns:
        matplotlib.figure.Figure: Figura creada
    """
    if not numerical_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No hay columnas numéricas\npara análisis de outliers',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig

    n_cols = min(len(numerical_cols), 4)  # Máximo 4 columnas
    selected_cols = numerical_cols[:n_cols]

    n_rows = (n_cols + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight='bold')

    for i, col in enumerate(selected_cols):
        row = i // 2
        col_pos = i % 2

        ax = axes[row][col_pos] if n_rows > 1 else axes[col_pos]

        # Box plot
        ax.boxplot(df[col].dropna(), vert=False, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', color='blue'),
                  medianprops=dict(color='red', linewidth=2))
        ax.set_title(f'Box Plot - {col}')
        ax.set_xlabel(col)
        ax.grid(True, alpha=0.3)

        # Calcular outliers usando IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        n_outliers = len(outliers)

        # Agregar información de outliers
        ax.text(0.02, 0.98, f'Outliers: {n_outliers}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # Si hay un subplot vacío, ocultarlo
    if n_cols % 2 != 0 and n_rows > 1:
        axes[-1][-1].set_visible(False)

    plt.tight_layout()
    return fig
