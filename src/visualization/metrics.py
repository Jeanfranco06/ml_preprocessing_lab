"""
Metrics Module
Funciones para calcular y mostrar métricas del dataset.
"""

import pandas as pd
import numpy as np
from .charts import create_dataset_overview_plot, create_preprocessing_comparison

def calculate_dataset_metrics(df):
    """
    Calcula métricas básicas del dataset.

    Args:
        df (pd.DataFrame): Dataset a analizar

    Returns:
        dict: Diccionario con métricas calculadas
    """
    metrics = {
        'basic_info': {
            'total_rows': df.shape[0],
            'total_columns': df.shape[1],
            'total_cells': df.shape[0] * df.shape[1],
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        },
        'data_types': {
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
            'boolean_columns': len(df.select_dtypes(include=['bool']).columns)
        },
        'missing_data': {
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'columns_with_missing': len(df.columns[df.isnull().any()]),
            'rows_with_missing': len(df[df.isnull().any(axis=1)])
        },
        'duplicates': {
            'total_duplicates': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / df.shape[0]) * 100
        }
    }

    # Estadísticas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        metrics['numeric_stats'] = {
            'columns': list(numeric_df.columns),
            'means': numeric_df.mean().to_dict(),
            'stds': numeric_df.std().to_dict(),
            'mins': numeric_df.min().to_dict(),
            'maxs': numeric_df.max().to_dict(),
            'medians': numeric_df.median().to_dict()
        }

    # Estadísticas categóricas
    categorical_df = df.select_dtypes(include=['object'])
    if not categorical_df.empty:
        cat_stats = {}
        for col in categorical_df.columns:
            value_counts = df[col].value_counts()
            cat_stats[col] = {
                'unique_values': len(value_counts),
                'most_common': value_counts.index[0],
                'most_common_count': value_counts.iloc[0],
                'most_common_percentage': (value_counts.iloc[0] / len(df)) * 100
            }
        metrics['categorical_stats'] = cat_stats

    return metrics

def calculate_preprocessing_metrics(original_df, processed_df, step_name):
    """
    Calcula métricas de comparación entre dataset original y procesado.

    Args:
        original_df (pd.DataFrame): Dataset original
        processed_df (pd.DataFrame): Dataset procesado
        step_name (str): Nombre del paso de procesamiento

    Returns:
        dict: Métricas de comparación
    """
    metrics = {
        'step': step_name,
        'changes': {
            'rows_change': processed_df.shape[0] - original_df.shape[0],
            'rows_change_percentage': ((processed_df.shape[0] - original_df.shape[0]) / original_df.shape[0]) * 100,
            'columns_change': processed_df.shape[1] - original_df.shape[1],
            'columns_change_percentage': ((processed_df.shape[1] - original_df.shape[1]) / original_df.shape[1]) * 100 if original_df.shape[1] > 0 else 0,
            'nulls_change': processed_df.isnull().sum().sum() - original_df.isnull().sum().sum(),
            'duplicates_change': processed_df.duplicated().sum() - original_df.duplicated().sum()
        },
        'final_state': {
            'total_rows': processed_df.shape[0],
            'total_columns': processed_df.shape[1],
            'total_nulls': processed_df.isnull().sum().sum(),
            'total_duplicates': processed_df.duplicated().sum(),
            'null_percentage': (processed_df.isnull().sum().sum() / (processed_df.shape[0] * processed_df.shape[1])) * 100,
            'duplicate_percentage': (processed_df.duplicated().sum() / processed_df.shape[0]) * 100
        }
    }

    return metrics

def create_metrics_summary(original_df, cleaned_df=None, encoded_df=None, normalized_df=None):
    """
    Crea un resumen completo de métricas de todo el proceso de preprocesamiento.

    Args:
        original_df, cleaned_df, encoded_df, normalized_df: DataFrames en cada etapa

    Returns:
        dict: Resumen completo de métricas
    """
    summary = {
        'original': calculate_dataset_metrics(original_df),
        'pipeline_steps': []
    }

    current_df = original_df

    if cleaned_df is not None:
        step_metrics = calculate_preprocessing_metrics(current_df, cleaned_df, "Limpieza de Datos")
        summary['pipeline_steps'].append(step_metrics)
        summary['cleaned'] = calculate_dataset_metrics(cleaned_df)
        current_df = cleaned_df

    if encoded_df is not None:
        step_metrics = calculate_preprocessing_metrics(current_df, encoded_df, "Codificación")
        summary['pipeline_steps'].append(step_metrics)
        summary['encoded'] = calculate_dataset_metrics(encoded_df)
        current_df = encoded_df

    if normalized_df is not None:
        step_metrics = calculate_preprocessing_metrics(current_df, normalized_df, "Normalización")
        summary['pipeline_steps'].append(step_metrics)
        summary['normalized'] = calculate_dataset_metrics(normalized_df)

    return summary

def display_metrics_dashboard(metrics, title="Dashboard de Métricas"):
    """
    Muestra un dashboard con las métricas calculadas.

    Args:
        metrics (dict): Diccionario de métricas
        title (str): Título del dashboard
    """
    import streamlit as st

    st.header(title)

    # Métricas básicas
    if 'basic_info' in metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total de Filas", f"{metrics['basic_info']['total_rows']:,}")
        with col2:
            st.metric("Total de Columnas", metrics['basic_info']['total_columns'])
        with col3:
            st.metric("Total de Celdas", f"{metrics['basic_info']['total_cells']:,}")
        with col4:
            st.metric("Uso de Memoria", ".2f")

    # Tipos de datos
    if 'data_types' in metrics:
        st.subheader("📊 Distribución de Tipos de Datos")
        types_df = pd.DataFrame([metrics['data_types']])
        st.dataframe(types_df.T, width='stretch')

    # Datos faltantes
    if 'missing_data' in metrics:
        st.subheader("❌ Análisis de Datos Faltantes")

        missing_cols = ['Total Missing', 'Missing %', 'Columnas con Missing', 'Filas con Missing']
        missing_vals = [
            metrics['missing_data']['total_missing'],
            ".2f",
            metrics['missing_data']['columns_with_missing'],
            metrics['missing_data']['rows_with_missing']
        ]

        missing_df = pd.DataFrame([missing_vals], columns=missing_cols)
        st.dataframe(missing_df, width='stretch')

    # Duplicados
    if 'duplicates' in metrics:
        st.subheader("🔄 Análisis de Duplicados")

        dup_cols = ['Total Duplicados', 'Duplicados %']
        dup_vals = [
            metrics['duplicates']['total_duplicates'],
            ".2f"
        ]

        dup_df = pd.DataFrame([dup_vals], columns=dup_cols)
        st.dataframe(dup_df, width='stretch')

def display_pipeline_summary(pipeline_summary):
    """
    Muestra un resumen del pipeline de preprocesamiento.

    Args:
        pipeline_summary (dict): Resumen del pipeline
    """
    import streamlit as st

    st.header("🔄 Resumen del Pipeline de Preprocesamiento")

    if 'pipeline_steps' in pipeline_summary:
        for i, step in enumerate(pipeline_summary['pipeline_steps'], 1):
            with st.expander(f"📍 Paso {i}: {step['step']}", expanded=(i==1)):
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📊 Cambios Realizados")
                    changes = step['changes']
                    changes_df = pd.DataFrame({
                        'Métrica': list(changes.keys()),
                        'Valor': [f"{v:,.0f}" if isinstance(v, (int, float)) and abs(v) >= 1 else ".2f" for v in changes.values()]
                    })
                    st.dataframe(changes_df, width='stretch')

                with col2:
                    st.subheader("📈 Estado Final")
                    final_state = step['final_state']
                    final_df = pd.DataFrame({
                        'Métrica': list(final_state.keys()),
                        'Valor': [f"{v:,.0f}" if isinstance(v, (int, float)) and abs(v) >= 1 else ".2f" for v in final_state.values()]
                    })
                    st.dataframe(final_df, width='stretch')

def calculate_model_readiness_score(df, target_col=None):
    """
    Calcula un puntaje de preparación para modelado.

    Args:
        df (pd.DataFrame): Dataset a evaluar
        target_col (str): Columna target si existe

    Returns:
        dict: Puntaje y recomendaciones
    """
    score = 0
    max_score = 100
    recommendations = []

    # Verificar valores nulos (20 puntos)
    null_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if null_percentage == 0:
        score += 20
        recommendations.append("✅ Excelente: No hay valores nulos")
    elif null_percentage < 5:
        score += 15
        recommendations.append("⚠️ Bueno: Pocos valores nulos, fácilmente manejables")
    else:
        score += 5
        recommendations.append("❌ Problema: Muchos valores nulos requieren atención")

    # Verificar duplicados (15 puntos)
    dup_percentage = (df.duplicated().sum() / df.shape[0]) * 100
    if dup_percentage == 0:
        score += 15
        recommendations.append("✅ Excelente: No hay duplicados")
    elif dup_percentage < 1:
        score += 10
        recommendations.append("⚠️ Aceptable: Pocos duplicados")
    else:
        score += 5
        recommendations.append("❌ Problema: Muchos duplicados")

    # Verificar balance de tipos de datos (15 puntos)
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    total_cols = df.shape[1]
    numeric_percentage = (numeric_cols / total_cols) * 100

    if 30 <= numeric_percentage <= 70:
        score += 15
        recommendations.append("✅ Bueno: Balance adecuado entre variables numéricas y categóricas")
    elif numeric_percentage > 80 or numeric_percentage < 20:
        score += 7
        recommendations.append("⚠️ Atención: Desbalance en tipos de variables")
    else:
        score += 10

    # Verificar dimensionalidad (20 puntos)
    if df.shape[0] > 1000 and df.shape[1] > 5:
        score += 20
        recommendations.append("✅ Excelente: Dataset de buen tamaño")
    elif df.shape[0] > 100 and df.shape[1] > 3:
        score += 15
        recommendations.append("⚠️ Aceptable: Dataset pequeño pero usable")
    else:
        score += 5
        recommendations.append("❌ Problema: Dataset muy pequeño")

    # Verificar target si existe (30 puntos)
    if target_col and target_col in df.columns:
        target_values = df[target_col].dropna()
        if len(target_values.unique()) > 1:
            score += 30
            recommendations.append("✅ Excelente: Variable target presente y variada")
        else:
            score += 10
            recommendations.append("❌ Problema: Variable target con un solo valor")
    else:
        score += 15
        recommendations.append("⚠️ Información: No se especificó variable target")

    readiness_level = "Bajo" if score < 40 else "Medio" if score < 70 else "Alto" if score < 90 else "Excelente"

    return {
        'score': score,
        'max_score': max_score,
        'percentage': (score / max_score) * 100,
        'readiness_level': readiness_level,
        'recommendations': recommendations
    }
