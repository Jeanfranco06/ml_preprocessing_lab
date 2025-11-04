# 🤖 Suite de Preprocesamiento de Datos ML

Una aplicación interactiva y completa de Streamlit para el preprocesamiento de datos en machine learning a través de múltiples conjuntos de datos. Esta aplicación demuestra técnicas profesionales de preprocesamiento con visualizaciones hermosas y capacidades de exportación.

## 🎯 Características

### 📊 **Soporte Multi-Dataset**
- **Titanic**: Preprocesamiento de predicción de supervivencia
- **Rendimiento Estudiantil**: Análisis de predicción de calificaciones
- **Iris**: Preprocesamiento de clasificación de especies

### 🔄 **Pipeline Completo de Preprocesamiento**
1. **Exploración Inicial**: Tipos de datos, valores faltantes, estadísticas descriptivas
2. **Limpieza de Datos**: Manejar nulos, eliminar duplicados, detectar outliers
3. **Codificación**: Codificación label y one-hot para variables categóricas
4. **Normalización**: Escalado estándar y min-max
5. **División Train/Test**: División configurable con estratificación
6. **Visualizaciones Avanzadas**: Insights específicos del dataset y correlaciones

### 🎨 **UI/UX Profesional**
- **Pestañas Interactivas**: Flujo de trabajo paso a paso
- **Métricas en Tiempo Real**: Actualizaciones en vivo de transformaciones
- **Visualizaciones Hermosas**: Gráficos Plotly y Seaborn
- **Diseño Responsivo**: Funciona en desktop y móvil
- **Tema Claro/Oscuro**: Selección de preferencia del usuario

### 💾 **Capacidades de Exportación**
- **Múltiples Formatos**: Exportación CSV, Excel, JSON
- **Resumen de Procesamiento**: Logs detallados de transformaciones
- **Generación de Código Pipeline**: Código Python automatizado
- **Descarga de Datos**: Datasets procesados en cualquier etapa

## 🚀 Inicio Rápido

### Prerrequisitos
- Python 3.8+
- Gestor de paquetes pip

### Instalación

1. **Clona o descarga el proyecto**
   ```bash
   cd ml-preprocessing-app
   ```

2. **Instala las dependencias**
   ```bash
   pip install -r requirements.txt
   ```

3. **Coloca los datasets en la carpeta `datasets/`**
   - `titanic.csv` - Dataset de supervivencia Titanic
   - `student-mat.csv` - Dataset de rendimiento estudiantil
   - El dataset Iris se carga automáticamente desde scikit-learn

4. **Ejecuta la aplicación**
   ```bash
   streamlit run app.py
   ```

5. **Abre tu navegador** en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
ml-preprocessing-app/
├── app.py                      # Aplicación principal Streamlit
├── requirements.txt            # Dependencias Python
├── README.md                   # Este archivo
├── datasets/                   # Almacenamiento de datasets
│   ├── titanic.csv            # Dataset Titanic (proporcionado por usuario)
│   └── student-mat.csv        # Dataset estudiantil (proporcionado por usuario)
├── utils/                      # Módulos utilitarios
│   ├── preprocessing.py       # Funciones de preprocesamiento
│   ├── visualization.py       # Funciones de gráficos
│   └── export.py              # Funciones de exportación
└── pages/                      # Páginas individuales por dataset
    ├── titanic.py             # Análisis específico Titanic
    ├── student.py             # Análisis rendimiento estudiantil
    └── iris.py                # Análisis dataset Iris
```

## 🎓 Valor Educativo

Esta aplicación sirve como herramienta integral de aprendizaje para:

- **Técnicas de Preprocesamiento**: Comprensión completa del pipeline ML
- **Ciencia de Datos Interactiva**: Experiencia práctica con datasets reales
- **Mejores Prácticas de Visualización**: Creación profesional de gráficos
- **Desarrollo Streamlit**: Creación de apps web para ciencia de datos
- **Exportación y Despliegue**: Hacer resultados de ciencia de datos compartibles

## 📊 Detalles de Datasets

### Dataset Titanic
- **Fuente**: Competencia Kaggle Titanic
- **Objetivo**: Predecir supervivencia de pasajeros
- **Características**: Demografía, info de tickets, detalles de cabina
- **Preprocesamiento**: Eliminar columnas irrelevantes, manejar edades/embarked faltantes, codificar categorías

### Dataset Rendimiento Estudiantil
- **Fuente**: Consumo de Alcohol Estudiantil Kaggle
- **Objetivo**: Predecir calificaciones finales (G3)
- **Características**: Demografía, antecedentes familiares, hábitos de estudio
- **Preprocesamiento**: One-hot encoding, manejar variables categóricas, normalizar calificaciones

### Dataset Iris
- **Fuente**: Dataset integrado de scikit-learn
- **Objetivo**: Clasificación de especies
- **Características**: Mediciones sépalo/pétalo
- **Preprocesamiento**: Estandarización, limpieza mínima necesaria

## 🛠️ Stack Técnico

- **Frontend**: Streamlit
- **Procesamiento de Datos**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualización**: matplotlib, seaborn, plotly
- **Exportación**: openpyxl, fpdf2
- **Componentes UI**: streamlit-extras

## 📈 Ejemplos de Uso

### Flujo de Trabajo Básico
1. Selecciona un dataset desde la barra lateral
2. Explora estadísticas iniciales y visualizaciones
3. Limpia datos (maneja valores faltantes, elimina duplicados)
4. Codifica variables categóricas
5. Normaliza características numéricas
6. Divide en conjuntos de entrenamiento/prueba
7. Visualiza insights avanzados
8. Exporta datos procesados y código generado

### Características Avanzadas
- **Controles Interactivos**: Ajusta parámetros de preprocesamiento
- **Comparación de Métodos**: Compara diferentes técnicas de normalización
- **Seguimiento de Progreso**: Estado de procesamiento en tiempo real
- **Generación de Código**: Exporta pipelines completos de preprocesamiento

## 🤝 Contribuyendo

Este es un proyecto educativo. Siéntete libre de:
- Agregar más datasets
- Implementar técnicas adicionales de preprocesamiento
- Mejorar visualizaciones
- Agregar más formatos de exportación
- Mejorar la UI/UX

## 📄 Licencia

Este proyecto es para fines educativos. Los datasets provienen de fuentes públicas con licencias apropiadas.

## 🙏 Agradecimientos

- **Streamlit** por el increíble framework de apps web
- **scikit-learn** por utilidades de machine learning
- **Kaggle** por datasets y comunidad
- **Plotly** y **Seaborn** por librerías de visualización

---

**¡Feliz aprendizaje! 🚀**

*Creado con ❤️ para educación en machine learning*
