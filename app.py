import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import os

# Page configuration
st.set_page_config(
    page_title="Procesamiento de Datasets ML",
    page_icon="📊",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .dataset-title {
        font-size: 1.6rem;
        font-weight: bold;
        color: #34495e;
        margin: 1.5rem 0 1rem 0;
    }
    .step-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1rem 0 0.5rem 0;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Procesamiento de Datasets en Machine Learning")

# Sidebar
st.sidebar.title("Configuración del Dataset")

# Dataset selection
dataset_options = {
    "Titanic": "titanic.csv",
    "Student Performance": "student-mat.csv",
    "Iris": "iris"
}

selected_dataset = st.sidebar.selectbox(
    "Seleccionar Dataset",
    list(dataset_options.keys())
)

# Load data function
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Iris":
        iris = load_iris()
        df = pd.DataFrame(
            data=np.c_[iris['data'], iris['target']],
            columns=iris['feature_names'] + ['target']
        )
        return df
    else:
        file_path = f"datasets/{dataset_options[dataset_name]}"
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            return None

# Load selected dataset
df = load_data(selected_dataset)

if df is not None:
    st.markdown(f'<h2 class="dataset-title">📊 Dataset: {selected_dataset}</h2>', unsafe_allow_html=True)

    # Create tabs for the 6 required steps
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Carga del Dataset",
        "2. Exploración Inicial",
        "3. Limpieza de Datos",
        "4. Codificación",
        "5. Normalización",
        "6. División Train/Test"
    ])

    # Step 1: Carga del Dataset
    with tab1:
        st.markdown('<h3 class="step-title">1. Carga del Dataset</h3>', unsafe_allow_html=True)

        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.write("**Dataset cargado exitosamente**")
        st.write(f"**Dimensiones:** {df.shape[0]} filas × {df.shape[1]} columnas")

        st.subheader("Primeras 5 filas:")
        st.dataframe(df.head())
        st.markdown('</div>', unsafe_allow_html=True)

    # Step 2: Exploración Inicial
    with tab2:
        st.markdown('<h3 class="step-title">2. Exploración Inicial</h3>', unsafe_allow_html=True)

        st.markdown('<div class="result-container">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Información del Dataset")
            st.write(f"**Tipo de datos por columna:**")
            for col, dtype in df.dtypes.items():
                st.write(f"- **{col}**: {dtype}")

        with col2:
            st.subheader("Estadísticas Descriptivas (.describe())")
            st.dataframe(df.describe())

        st.subheader("Valores Nulos")
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            st.dataframe(null_counts[null_counts > 0].to_frame('Valores Nulos'))
        else:
            st.write("✅ No se encontraron valores nulos")

        st.subheader("Tipos de Datos")
        st.dataframe(df.dtypes.to_frame('Tipo de Dato'))

        st.markdown('</div>', unsafe_allow_html=True)

    # Step 3: Limpieza de Datos
    with tab3:
        st.markdown('<h3 class="step-title">3. Limpieza de Datos</h3>', unsafe_allow_html=True)

        # Remove irrelevant columns for Titanic
        df_clean = df.copy()
        if selected_dataset == "Titanic":
            columns_to_remove = ['Name', 'Ticket', 'Cabin', 'PassengerId']
            existing_cols = [col for col in columns_to_remove if col in df_clean.columns]
            if existing_cols:
                df_clean = df_clean.drop(existing_cols, axis=1)
                st.write(f"✅ Columnas eliminadas: {existing_cols}")

        # Handle missing values
        if df_clean.isnull().sum().sum() > 0:
            # Fill numerical columns with mean
            numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)

            # Fill categorical columns with mode
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

            st.write("✅ Valores nulos manejados (numéricos: media, categóricos: moda)")

        # Remove duplicates
        initial_rows = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        final_rows = df_clean.shape[0]
        duplicates_removed = initial_rows - final_rows

        if duplicates_removed > 0:
            st.write(f"✅ Duplicados eliminados: {duplicates_removed}")
        else:
            st.write("✅ No se encontraron duplicados")

        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.subheader("Resultado de la Limpieza")
        st.write(f"**Dimensiones finales:** {df_clean.shape[0]} filas × {df_clean.shape[1]} columnas")
        st.dataframe(df_clean.head())
        st.markdown('</div>', unsafe_allow_html=True)

    # Step 4: Codificación
    with tab4:
        st.markdown('<h3 class="step-title">4. Codificación de Variables Categóricas</h3>', unsafe_allow_html=True)

        df_encoded = df_clean.copy()

        # Identify categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns

        if len(categorical_cols) > 0:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()

            encoding_map = {}
            for col in categorical_cols:
                original_values = df_encoded[col].unique()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                encoded_values = df_encoded[col].unique()
                encoding_map[col] = dict(zip(original_values, encoded_values))

            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.subheader("Mapeo de Codificación")
            for col, mapping in encoding_map.items():
                st.write(f"**Columna: {col}**")
                for original, encoded in mapping.items():
                    st.write(f"  {original} → {encoded}")
                st.write("---")

            st.subheader("Dataset Codificado")
            st.dataframe(df_encoded.head())
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            df_encoded = df_clean.copy()
            st.write("✅ No se encontraron variables categóricas para codificar")

    # Step 5: Normalización
    with tab5:
        st.markdown('<h3 class="step-title">5. Normalización/Estandarización</h3>', unsafe_allow_html=True)

        # Identify numerical columns (excluding target)
        target_col = 'Survived' if selected_dataset == 'Titanic' else ('G3' if selected_dataset == 'Student Performance' else 'target')
        numerical_cols = [col for col in df_encoded.select_dtypes(include=[np.number]).columns if col != target_col]

        if numerical_cols:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

            df_normalized = df_encoded.copy()
            df_normalized[numerical_cols] = scaler.fit_transform(df_normalized[numerical_cols])

            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.subheader("Normalización Aplicada (Standard Scaler)")
            st.write(f"**Columnas normalizadas:** {numerical_cols}")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Antes de normalización:**")
                st.dataframe(df_encoded[numerical_cols].describe())

            with col2:
                st.write("**Después de normalización:**")
                st.dataframe(df_normalized[numerical_cols].describe())

            st.subheader("Dataset Normalizado")
            st.dataframe(df_normalized.head())
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            df_normalized = df_encoded.copy()
            st.write("✅ No se encontraron columnas numéricas para normalizar")

    # Step 6: División Train/Test
    with tab6:
        st.markdown('<h3 class="step-title">6. División en Conjuntos de Entrenamiento y Prueba</h3>', unsafe_allow_html=True)

        from sklearn.model_selection import train_test_split

        # Define target column
        if selected_dataset == "Titanic":
            target_col = "Survived"
            test_size = 0.3  # 70% train, 30% test as per requirements
        elif selected_dataset == "Student Performance":
            target_col = "G3"
            test_size = 0.2  # 80% train, 20% test as per requirements
        else:  # Iris
            target_col = "target"
            test_size = 0.3  # 70% train, 30% test as per requirements

        if target_col in df_normalized.columns:
            X = df_normalized.drop(target_col, axis=1)
            y = df_normalized[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.subheader("División Train/Test Completada")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Conjunto de Entrenamiento:**")
                st.write(f"X_train: {X_train.shape[0]} filas × {X_train.shape[1]} columnas")
                st.write(f"y_train: {y_train.shape[0]} filas")

            with col2:
                st.write("**Conjunto de Prueba:**")
                st.write(f"X_test: {X_test.shape[0]} filas × {X_test.shape[1]} columnas")
                st.write(f"y_test: {y_test.shape[0]} filas")

            # Show first 5 processed records as required
            st.subheader("Primeros 5 registros procesados:")
            processed_df = X_train.copy()
            processed_df[target_col] = y_train
            st.dataframe(processed_df.head())

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.error(f"Columna target '{target_col}' no encontrada en el dataset")

else:
    st.error("Dataset no encontrado. Asegúrate de que los archivos estén en la carpeta 'datasets/'")
