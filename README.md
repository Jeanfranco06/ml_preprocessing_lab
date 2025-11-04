# Actividad Individual: Procesamiento de Datasets en Machine Learning

Aplicación Streamlit para el procesamiento de datos según los requerimientos de la actividad individual.

## Requisitos del Sistema
- Python 3.8+
- pip

## Instalación y Ejecución

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Ejecutar la aplicación:
```bash
streamlit run app.py
```

3. Abrir en navegador: `http://localhost:8501`

## Funcionalidades

La aplicación implementa las siguientes etapas de procesamiento para cada dataset:

### Etapas del Pipeline
1. **Carga del dataset**
2. **Exploración inicial** (info, describe, nulls, tipos de datos)
3. **Limpieza de datos** (valores nulos, duplicados, outliers)
4. **Codificación de variables categóricas**
5. **Normalización/estandarización**
6. **División en conjuntos de entrenamiento y prueba**

### Datasets Soportados
- **Titanic**: Predicción de supervivencia
- **Student Performance**: Predicción de calificaciones finales
- **Iris**: Clasificación de especies

## Estructura del Código

- `app.py`: Aplicación principal
- `utils/preprocessing.py`: Funciones de preprocesamiento
- `utils/visualization.py`: Funciones de visualización
- `utils/export.py`: Funciones de exportación
- `datasets/`: Archivos de datos
