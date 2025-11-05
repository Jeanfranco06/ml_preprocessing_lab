# 🤖 ML Preprocessing Lab

**Suite Avanzado de Preprocesamiento y Entrenamiento para Machine Learning**

Una aplicación profesional de Streamlit que implementa el pipeline completo de preprocesamiento de datos y **entrenamiento automático de modelos de ML** según los requerimientos de la **Actividad Individual** del curso de Machine Learning.

[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg)](https://python.org)

## 🎯 Características Principales

### 📊 **Pipeline Completo de 6 Etapas**
1. **📥 Carga del Dataset** - Importación automática desde archivos CSV
2. **🔍 Exploración Inicial** - Análisis estadístico completo (.info(), .describe(), nulos, tipos)
3. **🧹 Limpieza de Datos** - Manejo de nulos, duplicados y outliers
4. **🔤 Codificación** - Transformación de variables categóricas (Label Encoding)
5. **📏 Normalización** - Estandarización con Standard Scaler
6. **✂️ División Train/Test** - Separación estratificada con proporciones exactas

### 🎨 **Interfaz Profesional**
- ✅ **Interfaz personalizada** sin navegación automática
- ✅ **Diseño responsive** con paleta de colores profesional
- ✅ **Visualizaciones interactivas** en tiempo real
- ✅ **Métricas en tiempo real** con tarjetas animadas
- ✅ **Exportación múltiple** (CSV, Excel, JSON, Reportes Markdown)
- ✅ **Código reutilizable** generado automáticamente

### 📂 **Datasets Incluidos**
| Dataset | Descripción | Tamaño | Proporción Train/Test |
|---------|-------------|--------|----------------------|
| **🚢 Titanic** | Predicción de supervivencia | 891 filas | 70% / 30% |
| **🎓 Student Performance** | Predicción de calificaciones | 395 filas | 80% / 20% |
| **🌸 Iris** | Clasificación de especies | 150 filas | 70% / 30% |

## 🚀 Inicio Rápido

### Prerrequisitos
- **Python 3.8+**
- **pip** (gestor de paquetes)

### Instalación

1. **Clona o descarga el proyecto**
   ```bash
   git clone https://github.com/Jeanfranco06/ml_preprocessing_lab.git
   cd ml_preprocessing_lab
   ```

2. **Instala las dependencias**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecuta la aplicación**
   ```bash
   streamlit run app.py
   ```

4. **Abre tu navegador** en `http://localhost:8501`

## 🚀 Despliegue en Streamlit Cloud

### Requisitos Previos
- **Cuenta en GitHub** con el repositorio del proyecto
- **Cuenta en Streamlit Cloud** (gratuita)

### Pasos para Desplegar

1. **Sube el código a GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Ve a [Streamlit Cloud](https://share.streamlit.io)**

3. **Conecta tu repositorio**
   - Haz clic en "New app"
   - Selecciona tu repositorio de GitHub
   - Configura:
     - **Repository**: `Jeanfranco06/ml_preprocessing_lab`
     - **Branch**: `main`
     - **Main file path**: `app.py`
     - **Python version**: `3.8` o superior

4. **Haz clic en "Deploy"**

5. **¡Tu app estará lista en minutos!**

### Archivos de Configuración para Despliegue

El proyecto incluye todos los archivos necesarios para Streamlit Cloud:

- ✅ **`app.py`** - Archivo principal de la aplicación
- ✅ **`requirements.txt`** - Todas las dependencias Python
- ✅ **`packages.txt`** - Dependencias del sistema (si es necesario)
- ✅ **`.streamlit/config.toml`** - Configuración de Streamlit
- ✅ **`datasets/`** - Datos incluidos en el repositorio

### Solución de Problemas Comunes

**Error de memoria**: Si la app se queda sin memoria, considera reducir el tamaño de los datasets o optimizar las visualizaciones.

**Tiempo de carga**: Las primeras cargas pueden ser lentas. Streamlit Cloud optimiza automáticamente las cargas posteriores.

**Dependencias faltantes**: Asegúrate de que todas las librerías estén en `requirements.txt`.

## 📋 Uso de la Aplicación

### Navegación Automática
La aplicación utiliza la **navegación automática de Streamlit**. En el menú lateral encontrarás:

1. **🏠 Inicio** - Información general y resumen del proyecto
2. **🚢 Titanic** - Pipeline completo para dataset Titanic
3. **🎓 Student Performance** - Pipeline completo para dataset estudiantil
4. **🌸 Iris** - Pipeline completo para dataset Iris

### Flujo de Trabajo Típico
1. **Selecciona un dataset** del menú lateral
2. **Navega por las 6 pestañas** en orden secuencial
3. **Revisa los resultados** en cada etapa del pipeline
4. **Visualiza las métricas** y estadísticas generadas
5. **Exporta los resultados** en múltiples formatos

## 🏗️ Arquitectura del Proyecto

```
ml_preprocessing_lab/
├── 📂 config/                        # Configuración del sistema
│   └── settings.yaml                 # Configuración en YAML
├── 📂 src/                            # Código fuente modular
│   ├── __init__.py
│   ├── 📂 config/                     # Sistema de configuración
│   │   ├── __init__.py
│   │   └── config_manager.py          # Gestor de configuración
│   ├── 📂 utils/                      # Utilidades y helpers
│   │   ├── __init__.py
│   │   ├── logger.py                  # Sistema de logging
│   │   └── helpers.py                 # Funciones auxiliares
│   ├── 📂 data/                       # Manejo de datos
│   │   ├── __init__.py
│   │   ├── loaders.py                 # Carga de datasets
│   │   └── preprocessing.py           # Preprocesamiento
│   ├── 📂 models/                     # Modelos de ML
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Entrenamiento de modelos
│   │   ├── evaluator.py               # Evaluación de modelos
│   │   └── comparer.py                # Comparación de modelos
│   ├── 📂 visualization/              # Visualización
│   │   ├── __init__.py
│   │   ├── charts.py                  # Gráficos y visualizaciones
│   │   └── metrics.py                 # Métricas y KPIs
│   └── 📂 export/                     # Exportación y reportes
│       ├── __init__.py
│       └── reports.py                 # Generación de reportes
├── 📄 app.py                          # Página principal de Streamlit
├── 📂 pages/                          # Páginas adicionales de Streamlit
│   ├── Titanic.py                     # Pipeline completo Titanic
│   ├── Student_Performance.py         # Pipeline Student Performance
│   └── Iris.py                        # Pipeline Iris
├── 📂 datasets/                       # Datos del proyecto
│   ├── 📂 raw/                        # Datos originales
│   │   ├── titanic.csv
│   │   ├── student-mat.csv
│   │   └── iris.csv
│   └── 📂 processed/                  # Datos procesados
├── 📂 tests/                          # Tests unitarios
├── 📂 logs/                           # Archivos de log
├── 📂 .streamlit/                     # Configuración de Streamlit
│   └── config.toml
├── 📂 assets/                         # Recursos estáticos
│   └── styles.css
├── 📄 requirements.txt                # Dependencias Python
├── 📄 .gitignore                      # Archivos ignorados por Git
├── 📄 README.md                       # Documentación
└── 📄 LICENSE                         # Licencia MIT
```

## 📊 Requerimientos Académicos Cumplidos

### ✅ **Etapas del Pipeline (PDF)**
- ✅ **Carga del dataset** desde archivos CSV
- ✅ **Exploración inicial** (.info(), .describe(), valores nulos, tipos de datos)
- ✅ **Limpieza de datos** (manejo de nulos, eliminación de duplicados, outliers)
- ✅ **Codificación de variables categóricas** (Label Encoding)
- ✅ **Normalización/estandarización** (Standard Scaler)
- ✅ **División train/test** (proporciones exactas: 70/30, 80/20, 70/30)
- ✅ **Primeros 5 registros procesados** mostrados en cada dataset

### ✅ **Características Técnicas**
- ✅ **Interfaz intuitiva** con navegación automática
- ✅ **Visualizaciones claras** de cada etapa
- ✅ **Métricas detalladas** en tiempo real
- ✅ **Exportación de resultados** en múltiples formatos
- ✅ **Código modular** y bien estructurado
- ✅ **Documentación completa** del proceso

## 🛠️ Tecnologías Utilizadas

### **Core**
- **Streamlit 1.28+** - Framework web para aplicaciones de ML
- **Pandas** - Manipulación y análisis de datos
- **NumPy** - Computación numérica
- **Scikit-learn** - Algoritmos de ML y preprocesamiento

### **Visualización**
- **Matplotlib** - Gráficos base
- **Seaborn** - Visualizaciones estadísticas
- **Plotly** - Gráficos interactivos

### **Exportación**
- **OpenPyXL** - Exportación a Excel

## 📈 Resultados Esperados

### **Titanic Dataset**
- **Input**: 891 filas × 12 columnas
- **Output**: Dataset limpio con ~800 filas × 9 columnas
- **Train/Test**: 623/268 filas (70%/30%)
- **Variables target**: Supervivencia (0/1)

### **Student Performance Dataset**
- **Input**: 395 filas × 33 columnas
- **Output**: Dataset limpio con ~390 filas × 30 columnas
- **Train/Test**: 316/79 filas (80%/20%)
- **Variables target**: Calificación final G3 (0-20)

### **Iris Dataset**
- **Input**: 150 filas × 6 columnas
- **Output**: Dataset limpio con 150 filas × 5 columnas
- **Train/Test**: 105/45 filas (70%/30%)
- **Variables target**: Especies (setosa, versicolor, virginica)

## 🤝 Contribuciones

Este proyecto está diseñado para fines educativos. Sugerencias de mejora:

- Agregar más datasets
- Implementar técnicas adicionales de preprocesamiento
- Mejorar las visualizaciones
- Agregar más formatos de exportación
- Optimizar el rendimiento

## 📄 Licencia

Este proyecto está bajo la **Licencia MIT**. Ver archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- **Streamlit** por el increíble framework
- **Scikit-learn** por las utilidades de ML
- **Kaggle** por los datasets
- **Comunidad de ML** por el conocimiento compartido

---
