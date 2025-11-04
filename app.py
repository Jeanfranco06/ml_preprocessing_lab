import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import os
from utils.preprocessing import *
from utils.visualization import *
from utils.export import *

# Page configuration
st.set_page_config(
    page_title="ML Data Preprocessing Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown('<div class="sidebar-header">🤖 ML Preprocessing Suite</div>', unsafe_allow_html=True)

# Dataset selection
dataset_options = {
    "Titanic": "titanic.csv",
    "Student Performance": "student-mat.csv",
    "Iris": "iris"
}

selected_dataset = st.sidebar.selectbox(
    "Select Dataset",
    list(dataset_options.keys()),
    help="Choose the dataset you want to analyze"
)

# Theme toggle
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)

# Project info
with st.sidebar.expander("ℹ️ Project Information"):
    st.markdown("""
    **Machine Learning Data Preprocessing**

    This application demonstrates comprehensive data preprocessing techniques across three different datasets:

    - **Titanic**: Survival prediction
    - **Student Performance**: Grade prediction
    - **Iris**: Classification task

    Each dataset goes through: Exploration → Cleaning → Encoding → Normalization → Split → Visualization
    """)

# Main content
st.markdown('<h1 class="main-header">Machine Learning Data Preprocessing Suite</h1>', unsafe_allow_html=True)

# Load data based on selection
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Iris":
        iris = load_iris()
        df = pd.DataFrame(
            data=np.c_[iris['data'], iris['target']],
            columns=iris['feature_names'] + ['target']
        )
        df['target'] = df['target'].astype(int)
        return df, "iris"
    else:
        file_path = f"datasets/{dataset_options[dataset_name]}"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            return df, dataset_name.lower().replace(" ", "_")
        else:
            st.error(f"Dataset file not found: {file_path}")
            st.info("Please place the dataset files in the 'datasets' folder.")
            return None, None

df, dataset_key = load_data(selected_dataset)

if df is not None:
    # Create tabs for preprocessing steps
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Initial Exploration",
        "🧹 Data Cleaning",
        "🔤 Encoding",
        "📏 Normalization",
        "✂️ Data Split",
        "📈 Advanced Visualizations"
    ])

    with tab1:
        st.header("📊 Initial Data Exploration")
        show_initial_exploration(df, dataset_key)

    with tab2:
        st.header("🧹 Data Cleaning")
        cleaned_df = show_data_cleaning(df, dataset_key)

    with tab3:
        st.header("🔤 Categorical Encoding")
        if cleaned_df is not None:
            encoded_df = show_encoding(cleaned_df, dataset_key)
        else:
            encoded_df = None

    with tab4:
        st.header("📏 Normalization/Standardization")
        if encoded_df is not None:
            normalized_df = show_normalization(encoded_df, dataset_key)
        else:
            normalized_df = None

    with tab5:
        st.header("✂️ Train/Test Split")
        if normalized_df is not None:
            show_data_split(normalized_df, dataset_key)
        else:
            st.warning("Complete previous steps first")

    with tab6:
        st.header("📈 Advanced Visualizations & Insights")
        if normalized_df is not None:
            show_advanced_visualizations(normalized_df, dataset_key)
        else:
            st.warning("Complete preprocessing steps first")

    # Export section
    st.markdown("---")
    st.header("💾 Export Results")
    show_export_options(df, cleaned_df, encoded_df, normalized_df, dataset_key)

else:
    st.warning("Please ensure dataset files are placed in the 'datasets' folder.")
    st.info("Expected files: titanic.csv, student-mat.csv")
