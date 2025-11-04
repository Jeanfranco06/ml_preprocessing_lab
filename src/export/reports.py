"""
Reports Module
Funciones para generar reportes y exportar resultados.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

def generate_preprocessing_report(original_df, cleaned_df=None, encoded_df=None,
                                normalized_df=None, dataset_key="dataset"):
    """
    Genera un reporte completo del proceso de preprocesamiento.

    Args:
        original_df, cleaned_df, encoded_df, normalized_df: DataFrames en cada etapa
        dataset_key (str): Identificador del dataset

    Returns:
        str: Reporte en formato Markdown
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# 📊 Reporte de Preprocesamiento de Datos

**Dataset:** {dataset_key.upper()}
**Fecha de Generación:** {timestamp}
**Herramienta:** ML Preprocessing Suite

---

## 📈 Resumen Ejecutivo

"""

    # Métricas principales
    original_rows, original_cols = original_df.shape
    original_nulls = original_df.isnull().sum().sum()

    report += f"""### 📊 Métricas del Dataset Original
- **Filas:** {original_rows:,}
- **Columnas:** {original_cols}
- **Valores nulos:** {original_nulls:,}
- **Porcentaje de nulos:** {(original_nulls / (original_rows * original_cols) * 100):.2f}%

"""

    # Resumen de etapas
    stages_summary = []

    if cleaned_df is not None:
        cleaned_rows, cleaned_cols = cleaned_df.shape
        cleaned_nulls = cleaned_df.isnull().sum().sum()
        stages_summary.append({
            'etapa': 'Limpieza',
            'filas': cleaned_rows,
            'columnas': cleaned_cols,
            'nulos': cleaned_nulls
        })

    if encoded_df is not None:
        encoded_rows, encoded_cols = encoded_df.shape
        encoded_nulls = encoded_df.isnull().sum().sum()
        stages_summary.append({
            'etapa': 'Codificación',
            'filas': encoded_rows,
            'columnas': encoded_cols,
            'nulos': encoded_nulls
        })

    if normalized_df is not None:
        normalized_rows, normalized_cols = normalized_df.shape
        normalized_nulls = normalized_df.isnull().sum().sum()
        stages_summary.append({
            'etapa': 'Normalización',
            'filas': normalized_rows,
            'columnas': normalized_cols,
            'nulos': normalized_nulls
        })

    if stages_summary:
        report += "### 🔄 Etapas del Pipeline\n\n"
        report += "| Etapa | Filas | Columnas | Valores Nulos |\n"
        report += "|-------|-------|----------|---------------|\n"

        for stage in stages_summary:
            report += f"| {stage['etapa']} | {stage['filas']:,} | {stage['columnas']} | {stage['nulos']:,} |\n"

        report += "\n"

    # Análisis detallado
    report += "## 📋 Análisis Detallado\n\n"

    # Estadísticas descriptivas
    report += "### 📊 Estadísticas Descriptivas\n\n"

    if normalized_df is not None:
        final_df = normalized_df
        stage_name = "Dataset Final (Normalizado)"
    elif encoded_df is not None:
        final_df = encoded_df
        stage_name = "Dataset Final (Codificado)"
    elif cleaned_df is not None:
        final_df = cleaned_df
        stage_name = "Dataset Final (Limpio)"
    else:
        final_df = original_df
        stage_name = "Dataset Original"

    numeric_cols = final_df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        report += f"#### {stage_name} - Variables Numéricas\n\n"
        report += "| Variable | Media | Desv. Estándar | Mínimo | Máximo | Mediana |\n"
        report += "|----------|-------|----------------|--------|--------|---------|\n"

        for col in numeric_cols[:10]:  # Limitar a 10 columnas para legibilidad
            stats = final_df[col].describe()
            report += f"| {col} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['50%']:.4f} |\n"

        report += "\n"

    # Variables categóricas
    categorical_cols = final_df.select_dtypes(include=['object']).columns

    if len(categorical_cols) > 0:
        report += f"#### {stage_name} - Variables Categóricas\n\n"

        for col in categorical_cols[:5]:  # Limitar a 5 columnas
            value_counts = final_df[col].value_counts()
            report += f"**{col}:**\n"
            for val, count in value_counts.head(5).items():
                percentage = (count / len(final_df)) * 100
                report += f"- {val}: {count:,} ({percentage:.1f}%)\n"
            report += "\n"

    # Recomendaciones
    report += "## 💡 Recomendaciones\n\n"

    recommendations = []

    # Verificar calidad de datos
    final_null_percentage = (final_df.isnull().sum().sum() / (final_df.shape[0] * final_df.shape[1])) * 100
    if final_null_percentage == 0:
        recommendations.append("✅ **Excelente calidad de datos**: No hay valores nulos restantes")
    elif final_null_percentage < 1:
        recommendations.append("⚠️ **Buena calidad**: Pocos valores nulos, aceptable para modelado")
    else:
        recommendations.append("❌ **Atención requerida**: Aún hay valores nulos que pueden afectar el modelado")

    # Verificar dimensionalidad
    if final_df.shape[0] > 1000:
        recommendations.append("✅ **Dataset adecuado**: Tamaño suficiente para entrenamiento confiable")
    elif final_df.shape[0] > 100:
        recommendations.append("⚠️ **Dataset pequeño**: Considerar técnicas de data augmentation")
    else:
        recommendations.append("❌ **Dataset muy pequeño**: Riesgo de overfitting")

    # Verificar balance de features
    numeric_percentage = (len(numeric_cols) / final_df.shape[1]) * 100
    if 20 <= numeric_percentage <= 80:
        recommendations.append("✅ **Balance adecuado**: Buena mezcla de variables numéricas y categóricas")
    else:
        recommendations.append("⚠️ **Desbalance de tipos**: Considerar transformación de variables")

    for rec in recommendations:
        report += f"- {rec}\n"

    report += "\n---\n"
    report += f"*Reporte generado automáticamente por ML Preprocessing Suite - {timestamp}*"

    return report

def export_to_csv(df, filename, output_dir="exports"):
    """
    Exporta un DataFrame a CSV.

    Args:
        df (pd.DataFrame): DataFrame a exportar
        filename (str): Nombre del archivo (sin extensión)
        output_dir (str): Directorio de salida

    Returns:
        str: Ruta del archivo exportado
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{filename}.csv")
    df.to_csv(file_path, index=False)
    return file_path

def export_to_excel(df, filename, output_dir="exports"):
    """
    Exporta un DataFrame a Excel.

    Args:
        df (pd.DataFrame): DataFrame a exportar
        filename (str): Nombre del archivo (sin extensión)
        output_dir (str): Directorio de salida

    Returns:
        str: Ruta del archivo exportado
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{filename}.xlsx")

    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Datos', index=False)

        # Agregar hoja con estadísticas
        if len(df.select_dtypes(include=[np.number]).columns) > 0:
            numeric_stats = df.describe()
            numeric_stats.to_excel(writer, sheet_name='Estadísticas')

    return file_path

def export_to_json(df, filename, output_dir="exports"):
    """
    Exporta un DataFrame a JSON.

    Args:
        df (pd.DataFrame): DataFrame a exportar
        filename (str): Nombre del archivo (sin extensión)
        output_dir (str): Directorio de salida

    Returns:
        str: Ruta del archivo exportado
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{filename}.json")

    # Convertir a formato JSON serializable
    json_data = df.to_dict('records')

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    return file_path

def generate_pipeline_code(dataset_key, preprocessing_steps):
    """
    Genera código Python reutilizable del pipeline aplicado.

    Args:
        dataset_key (str): Identificador del dataset
        preprocessing_steps (list): Lista de pasos aplicados

    Returns:
        str: Código Python generado
    """
    code = f'''"""
Pipeline de Preprocesamiento para {dataset_key.upper()}
Generado automáticamente por ML Preprocessing Suite
Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_{dataset_key}_data(df):
    """
    Aplica el pipeline completo de preprocesamiento.

    Args:
        df (pd.DataFrame): Dataset original

    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
'''

    # Agregar pasos de preprocesamiento
    steps_code = []

    if "limpieza" in preprocessing_steps:
        steps_code.append('''
    # 1. Limpieza de datos
    # Eliminar columnas irrelevantes
    columns_to_drop = []  # Agregar columnas según sea necesario
    df = df.drop(columns_to_drop, axis=1, errors='ignore')

    # Manejar valores nulos
    # Numéricos: media
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    # Categóricos: moda
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Eliminar duplicados
    df = df.drop_duplicates()
''')

    if "codificación" in preprocessing_steps:
        steps_code.append('''
    # 2. Codificación de variables categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()

    encoding_map = {}
    for col in categorical_cols:
        original_values = df[col].unique()
        df[col] = le.fit_transform(df[col].astype(str))
        encoded_values = df[col].unique()
        encoding_map[col] = dict(zip(original_values, encoded_values))
''')

    if "normalización" in preprocessing_steps:
        steps_code.append('''
    # 3. Normalización de variables numéricas
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Excluir columna target si existe
    target_cols = ['target', 'survived', 'g3']  # Posibles nombres de target
    feature_cols = [col for col in numeric_cols if col.lower() not in target_cols]

    if feature_cols:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
''')

    # Agregar código de división train/test
    steps_code.append('''
    # 4. División train/test
    # Identificar columna target
    possible_targets = ['target', 'survived', 'g3']
    target_col = None

    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col:
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test, scaler if 'scaler' in locals() else None
    else:
        # Si no hay target, dividir todo el dataset
        X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)
        return X_train, X_test, None, None, scaler if 'scaler' in locals() else None
''')

    # Unir todos los pasos
    code += '\n'.join(steps_code)

    code += '''
# Ejemplo de uso:
# df = pd.read_csv('tu_dataset.csv')
# X_train, X_test, y_train, y_test, scaler = preprocess_tu_dataset_data(df)
'''

    return code

def create_comprehensive_report(original_df, cleaned_df=None, encoded_df=None,
                              normalized_df=None, dataset_key="dataset"):
    """
    Crea un reporte comprehensivo con múltiples formatos.

    Args:
        original_df, cleaned_df, encoded_df, normalized_df: DataFrames
        dataset_key (str): Identificador del dataset

    Returns:
        dict: Diccionario con rutas de archivos generados
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{dataset_key}_preprocessing_{timestamp}"

    generated_files = {}

    # Generar reporte Markdown
    markdown_report = generate_preprocessing_report(
        original_df, cleaned_df, encoded_df, normalized_df, dataset_key
    )

    # Exportar a diferentes formatos
    if normalized_df is not None:
        final_df = normalized_df
    elif encoded_df is not None:
        final_df = encoded_df
    elif cleaned_df is not None:
        final_df = cleaned_df
    else:
        final_df = original_df

    # CSV
    csv_path = export_to_csv(final_df, f"{base_filename}_final", "exports")
    generated_files['csv'] = csv_path

    # Excel
    excel_path = export_to_excel(final_df, f"{base_filename}_report", "exports")
    generated_files['excel'] = excel_path

    # JSON
    json_path = export_to_json(final_df, f"{base_filename}_data", "exports")
    generated_files['json'] = json_path

    # Markdown report
    report_path = os.path.join("exports", f"{base_filename}_report.md")
    os.makedirs("exports", exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    generated_files['markdown_report'] = report_path

    # Pipeline code
    preprocessing_steps = []
    if cleaned_df is not None:
        preprocessing_steps.append("limpieza")
    if encoded_df is not None:
        preprocessing_steps.append("codificación")
    if normalized_df is not None:
        preprocessing_steps.append("normalización")

    pipeline_code = generate_pipeline_code(dataset_key, preprocessing_steps)
    code_path = os.path.join("exports", f"{base_filename}_pipeline.py")
    with open(code_path, 'w', encoding='utf-8') as f:
        f.write(pipeline_code)
    generated_files['pipeline_code'] = code_path

    return generated_files
