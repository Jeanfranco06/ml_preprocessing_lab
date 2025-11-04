"""
ML Preprocessing Lab - Single Page Application
A clean, simple Streamlit app for machine learning preprocessing pipelines.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="ML Preprocessing Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🔬 ML Preprocessing Lab")
st.sidebar.markdown("---")

# Dataset selection
dataset_options = {
    "🏠 Home": "home",
    "🚢 Titanic Dataset": "titanic",
    "📚 Student Performance": "student_performance",
    "🌸 Iris Classification": "iris"
}

selected_page = st.sidebar.selectbox(
    "Choose Dataset:",
    list(dataset_options.keys()),
    index=0
)

dataset_key = dataset_options[selected_page]

# Helper functions
def load_titanic_data():
    """Load Titanic dataset from CSV file."""
    data_dir = Path(__file__).parent / "datasets" / "raw"
    file_path = data_dir / "titanic.csv"

    if not file_path.exists():
        st.error(f"Dataset file not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def load_student_performance_data():
    """Load Student Performance dataset from CSV file."""
    data_dir = Path(__file__).parent / "datasets" / "raw"
    file_path = data_dir / "student-mat.csv"

    if not file_path.exists():
        st.error(f"Dataset file not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, sep=';')  # Student dataset uses semicolon separator
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def load_iris_data():
    """Load Iris dataset from CSV file."""
    data_dir = Path(__file__).parent / "datasets" / "raw"
    file_path = data_dir / "iris.csv"

    if not file_path.exists():
        st.error(f"Dataset file not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Main content
if dataset_key == "home":
    # Home page
    st.markdown('<div class="main-header">🔬 ML Preprocessing Lab</div>', unsafe_allow_html=True)

    st.markdown("""
    ## Welcome to the Machine Learning Preprocessing Suite

    This application provides a complete 6-stage preprocessing pipeline for machine learning datasets:

    ### 📊 **Available Datasets:**
    - **🚢 Titanic**: Passenger survival prediction
    - **📚 Student Performance**: Academic performance prediction
    - **🌸 Iris**: Flower species classification

    ### 🔄 **Preprocessing Pipeline:**
    1. **📊 Data Loading** - Load and validate datasets
    2. **🔍 Data Exploration** - Analyze distributions and patterns
    3. **🧹 Data Cleaning** - Handle missing values and duplicates
    4. **🔤 Encoding** - Convert categorical variables
    5. **📏 Normalization** - Scale numerical features
    6. **✂️ Train/Test Split** - Prepare data for modeling

    ### 🚀 **Getting Started:**
    Select a dataset from the sidebar to begin preprocessing your data!
    """)

    # Quick stats
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Datasets Available", "3")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Preprocessing Stages", "6")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("ML Ready", "✅")
        st.markdown('</div>', unsafe_allow_html=True)

elif dataset_key == "titanic":
    # Titanic Dataset Page
    st.markdown('<div class="main-header">🚢 Titanic Dataset - Survival Prediction</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 📋 Dataset Overview
    The Titanic dataset contains information about passengers on the RMS Titanic,
    including demographic data and survival outcomes. This page provides a complete
    6-stage preprocessing pipeline.
    """)

    # Stage 1: Data Loading
    st.markdown('<div class="section-header">📊 Stage 1: Data Loading</div>', unsafe_allow_html=True)

    if st.button("🔄 Load Titanic Dataset", type="primary"):
        df = load_titanic_data()

        if df is not None:
            st.session_state.titanic_raw = df.copy()
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Passengers", f"{df.shape[0]:,}")
            with col2:
                survived = df['Survived'].sum()
                st.metric("Survived", f"{survived:,}")
            with col3:
                survival_rate = (survived / df.shape[0]) * 100
                st.metric("Survival Rate", f"{survival_rate:.1f}%")

            st.dataframe(df.head(), use_container_width=True)
        else:
            st.error("❌ Failed to load dataset")

    # Check if data is loaded
    if 'titanic_raw' not in st.session_state:
        st.info("👆 Please load the dataset first to continue with preprocessing.")
    else:
        df = st.session_state.titanic_raw

        # Create tabs for preprocessing stages
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔍 Exploration",
            "🧹 Cleaning",
            "🔤 Encoding",
            "📏 Normalization",
            "✂️ Train/Test Split",
            "📊 Results Summary"
        ])

        # Stage 2: Data Exploration
        with tab1:
            st.markdown("### 🔍 Stage 2: Data Exploration")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Data Types")
                dtypes_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str)
                })
                st.dataframe(dtypes_df, use_container_width=True)

            with col2:
                st.subheader("Missing Values")
                null_counts = df.isnull().sum()
                null_df = pd.DataFrame({
                    'Column': null_counts.index,
                    'Missing': null_counts.values,
                    'Percentage': (null_counts / len(df) * 100).round(2)
                })
                st.dataframe(null_df[null_df['Missing'] > 0], use_container_width=True)

            # Basic statistics
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)

            # Visualizations
            st.subheader("Survival Distribution")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Survival count
            survival_counts = df['Survived'].value_counts()
            ax1.bar(['Did not survive', 'Survived'], survival_counts.values,
                   color=['#e74c3c', '#27ae60'])
            ax1.set_title('Survival Count')
            ax1.set_ylabel('Number of Passengers')

            # Survival by class
            class_survival = df.groupby('Pclass')['Survived'].mean()
            ax2.bar(['Class 1', 'Class 2', 'Class 3'], class_survival.values,
                   color=['#3498db', '#f39c12', '#e74c3c'])
            ax2.set_title('Survival Rate by Class')
            ax2.set_ylabel('Survival Rate')
            ax2.set_ylim(0, 1)

            plt.tight_layout()
            st.pyplot(fig)

        # Stage 3: Data Cleaning
        with tab2:
            st.markdown("### 🧹 Stage 3: Data Cleaning")

            # Remove irrelevant columns
            irrelevant_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
            cols_to_remove = [col for col in irrelevant_cols if col in df.columns]

            df_clean = df.drop(cols_to_remove, axis=1)

            # Handle missing values
            # Age: fill with median
            if 'Age' in df_clean.columns and df_clean['Age'].isnull().sum() > 0:
                df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())

            # Embarked: fill with mode
            if 'Embarked' in df_clean.columns and df_clean['Embarked'].isnull().sum() > 0:
                df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])

            # Remove duplicates
            initial_rows = df_clean.shape[0]
            df_clean = df_clean.drop_duplicates()
            final_rows = df_clean.shape[0]

            # Store cleaned data
            st.session_state.titanic_clean = df_clean

            # Show results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Before Cleaning")
                st.metric("Rows", df.shape[0])
                st.metric("Columns", df.shape[1])
                st.metric("Missing Values", df.isnull().sum().sum())

            with col2:
                st.subheader("After Cleaning")
                st.metric("Rows", df_clean.shape[0])
                st.metric("Columns", df_clean.shape[1])
                st.metric("Missing Values", df_clean.isnull().sum().sum())

            if initial_rows > final_rows:
                st.success(f"✅ Removed {initial_rows - final_rows} duplicate rows")

            st.dataframe(df_clean.head(), use_container_width=True)

        # Stage 4: Encoding
        with tab3:
            st.markdown("### 🔤 Stage 4: Categorical Encoding")

            if 'titanic_clean' not in st.session_state:
                st.warning("Please complete data cleaning first")
            else:
                df_clean = st.session_state.titanic_clean

                # Identify categorical columns
                categorical_cols = ['Sex', 'Embarked']
                categorical_cols = [col for col in categorical_cols if col in df_clean.columns]

                if categorical_cols:
                    df_encoded = df_clean.copy()
                    encoding_map = {}

                    for col in categorical_cols:
                        le = LabelEncoder()
                        original_values = df_encoded[col].unique()
                        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                        encoded_values = df_encoded[col].unique()
                        encoding_map[col] = dict(zip(original_values, encoded_values))

                    # Store encoded data
                    st.session_state.titanic_encoded = df_encoded
                    st.session_state.titanic_encoding_map = encoding_map

                    # Show encoding mapping
                    st.subheader("Encoding Mapping")
                    for col, mapping in encoding_map.items():
                        st.write(f"**{col}:**")
                        for original, encoded in mapping.items():
                            st.write(f"  `{original}` → `{encoded}`")
                        st.write("---")

                    st.dataframe(df_encoded.head(), use_container_width=True)
                    st.success("✅ Categorical variables encoded successfully")
                else:
                    st.session_state.titanic_encoded = df_clean
                    st.info("No categorical columns found to encode")

        # Stage 5: Normalization
        with tab4:
            st.markdown("### 📏 Stage 5: Feature Normalization")

            if 'titanic_encoded' not in st.session_state:
                st.warning("Please complete encoding first")
            else:
                df_encoded = st.session_state.titanic_encoded

                # Identify numerical columns (excluding target)
                numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
                numerical_cols = [col for col in numerical_cols if col in df_encoded.columns and col != 'Survived']

                if numerical_cols:
                    df_normalized = df_encoded.copy()

                    # Apply Standard Scaler
                    scaler = StandardScaler()
                    df_normalized[numerical_cols] = scaler.fit_transform(df_normalized[numerical_cols])

                    # Store normalized data and scaler
                    st.session_state.titanic_normalized = df_normalized
                    st.session_state.titanic_scaler = scaler

                    # Show comparison
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Before Normalization")
                        st.dataframe(df_encoded[numerical_cols].describe())

                    with col2:
                        st.subheader("After Normalization")
                        st.dataframe(df_normalized[numerical_cols].describe())

                    st.success("✅ Numerical features normalized with Standard Scaler")
                else:
                    st.session_state.titanic_normalized = df_encoded
                    st.info("No numerical columns found to normalize")

        # Stage 6: Train/Test Split
        with tab5:
            st.markdown("### ✂️ Stage 6: Train/Test Split")

            if 'titanic_normalized' not in st.session_state:
                st.warning("Please complete normalization first")
            else:
                df_normalized = st.session_state.titanic_normalized

                # Prepare features and target
                X = df_normalized.drop('Survived', axis=1)
                y = df_normalized['Survived']

                # Split data (70% train, 30% test)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )

                # Store split data
                st.session_state.titanic_X_train = X_train
                st.session_state.titanic_X_test = X_test
                st.session_state.titanic_y_train = y_train
                st.session_state.titanic_y_test = y_test

                # Show results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Training Set")
                    st.metric("X_train shape", f"{X_train.shape[0]} × {X_train.shape[1]}")
                    st.metric("y_train shape", f"{y_train.shape[0]}")

                    # Class distribution in train
                    train_survival = y_train.value_counts().sort_index()
                    st.write("**Survival distribution:**")
                    st.write(f"- Did not survive: {train_survival.iloc[0]} ({train_survival.iloc[0]/len(y_train)*100:.1f}%)")
                    st.write(f"- Survived: {train_survival.iloc[1]} ({train_survival.iloc[1]/len(y_train)*100:.1f}%)")

                with col2:
                    st.subheader("Test Set")
                    st.metric("X_test shape", f"{X_test.shape[0]} × {X_test.shape[1]}")
                    st.metric("y_test shape", f"{y_test.shape[0]}")

                    # Class distribution in test
                    test_survival = y_test.value_counts().sort_index()
                    st.write("**Survival distribution:**")
                    st.write(f"- Did not survive: {test_survival.iloc[0]} ({test_survival.iloc[0]/len(y_test)*100:.1f}%)")
                    st.write(f"- Survived: {test_survival.iloc[1]} ({test_survival.iloc[1]/len(y_test)*100:.1f}%)")

                st.success("✅ Data split completed with stratification")

        # Results Summary
        with tab6:
            st.markdown("### 📊 Preprocessing Results Summary")

            if all(key in st.session_state for key in ['titanic_X_train', 'titanic_X_test', 'titanic_y_train', 'titanic_y_test']):
                st.markdown('<div class="success-message">🎉 All preprocessing stages completed successfully!</div>', unsafe_allow_html=True)

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Original Rows", f"{st.session_state.titanic_raw.shape[0]:,}")
                with col2:
                    st.metric("Final Rows", f"{st.session_state.titanic_X_train.shape[0] + st.session_state.titanic_X_test.shape[0]:,}")
                with col3:
                    st.metric("Features", st.session_state.titanic_X_train.shape[1])
                with col4:
                    st.metric("Train/Test Ratio", "70/30")

                # Show final datasets
                st.subheader("Training Data Preview")
                train_preview = st.session_state.titanic_X_train.copy()
                train_preview['Survived'] = st.session_state.titanic_y_train
                st.dataframe(train_preview.head(), use_container_width=True)

                st.subheader("Test Data Preview")
                test_preview = st.session_state.titanic_X_test.copy()
                test_preview['Survived'] = st.session_state.titanic_y_test
                st.dataframe(test_preview.head(), use_container_width=True)

                # Export options
                st.subheader("💾 Export Processed Data")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("📊 Export Training Data"):
                        train_df = st.session_state.titanic_X_train.copy()
                        train_df['Survived'] = st.session_state.titanic_y_train
                        csv = train_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Training CSV",
                            csv,
                            "titanic_train_processed.csv",
                            "text/csv"
                        )

                with col2:
                    if st.button("📊 Export Test Data"):
                        test_df = st.session_state.titanic_X_test.copy()
                        test_df['Survived'] = st.session_state.titanic_y_test
                        csv = test_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Test CSV",
                            csv,
                            "titanic_test_processed.csv",
                            "text/csv"
                        )
            else:
                st.warning("Please complete all preprocessing stages to see the summary")

elif dataset_key == "student_performance":
    # Student Performance Dataset Page
    st.markdown('<div class="main-header">📚 Student Performance Dataset - Grade Prediction</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 📋 Dataset Overview
    The Student Performance dataset contains information about students' academic performance,
    including demographic data, family background, and final grades. This page provides a complete
    6-stage preprocessing pipeline for predicting student grades (G3).
    """)

    # Stage 1: Data Loading
    st.markdown('<div class="section-header">📊 Stage 1: Data Loading</div>', unsafe_allow_html=True)

    if st.button("🔄 Load Student Performance Dataset", type="primary"):
        df = load_student_performance_data()

        if df is not None:
            st.session_state.student_raw = df.copy()
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Students", f"{df.shape[0]:,}")
            with col2:
                avg_grade = df['G3'].mean()
                st.metric("Average Final Grade", f"{avg_grade:.1f}")
            with col3:
                passing_rate = (df['G3'] >= 10).mean() * 100  # Assuming 10 is passing grade
                st.metric("Passing Rate", f"{passing_rate:.1f}%")

            st.dataframe(df.head(), use_container_width=True)
        else:
            st.error("❌ Failed to load dataset")

    # Check if data is loaded
    if 'student_raw' not in st.session_state:
        st.info("👆 Please load the dataset first to continue with preprocessing.")
    else:
        df = st.session_state.student_raw

        # Create tabs for preprocessing stages
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔍 Exploration",
            "🧹 Cleaning",
            "🔤 Encoding",
            "📏 Normalization",
            "✂️ Train/Test Split",
            "📊 Results Summary"
        ])

        # Stage 2: Data Exploration
        with tab1:
            st.markdown("### 🔍 Stage 2: Data Exploration")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Data Types")
                dtypes_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str)
                })
                st.dataframe(dtypes_df, use_container_width=True)

            with col2:
                st.subheader("Missing Values")
                null_counts = df.isnull().sum()
                if null_counts.sum() == 0:
                    st.success("✅ No missing values found")
                else:
                    null_df = pd.DataFrame({
                        'Column': null_counts.index,
                        'Missing': null_counts.values,
                        'Percentage': (null_counts / len(df) * 100).round(2)
                    })
                    st.dataframe(null_df[null_df['Missing'] > 0], use_container_width=True)

            # Basic statistics
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)

            # Grade distribution
            st.subheader("Grade Distribution")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

            # G1 distribution
            ax1.hist(df['G1'], bins=20, alpha=0.7, color='#1f77b4', edgecolor='black')
            ax1.set_title('First Period Grade (G1)')
            ax1.set_xlabel('Grade')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)

            # G2 distribution
            ax2.hist(df['G2'], bins=20, alpha=0.7, color='#ff7f0e', edgecolor='black')
            ax2.set_title('Second Period Grade (G2)')
            ax2.set_xlabel('Grade')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)

            # G3 distribution
            ax3.hist(df['G3'], bins=20, alpha=0.7, color='#2ca02c', edgecolor='black')
            ax3.set_title('Final Grade (G3) - Target')
            ax3.set_xlabel('Grade')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)

            # Study time vs Final grade
            study_grades = df.groupby('studytime')['G3'].mean()
            ax4.bar(study_grades.index, study_grades.values, color='#d62728', alpha=0.7)
            ax4.set_title('Average Final Grade by Study Time')
            ax4.set_xlabel('Study Time (hours)')
            ax4.set_ylabel('Average Final Grade')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

        # Stage 3: Data Cleaning
        with tab2:
            st.markdown("### 🧹 Stage 3: Data Cleaning")

            # For student performance dataset, minimal cleaning needed
            # No irrelevant columns to remove, no missing values
            df_clean = df.copy()

            # Remove duplicates if any
            initial_rows = df_clean.shape[0]
            df_clean = df_clean.drop_duplicates()
            final_rows = df_clean.shape[0]

            # Store cleaned data
            st.session_state.student_clean = df_clean

            # Show results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Before Cleaning")
                st.metric("Rows", df.shape[0])
                st.metric("Columns", df.shape[1])
                st.metric("Missing Values", df.isnull().sum().sum())

            with col2:
                st.subheader("After Cleaning")
                st.metric("Rows", df_clean.shape[0])
                st.metric("Columns", df_clean.shape[1])
                st.metric("Missing Values", df_clean.isnull().sum().sum())

            if initial_rows > final_rows:
                st.success(f"✅ Removed {initial_rows - final_rows} duplicate rows")
            else:
                st.success("✅ Dataset was already clean - no duplicates or missing values")

            st.dataframe(df_clean.head(), use_container_width=True)

        # Stage 4: Encoding
        with tab3:
            st.markdown("### 🔤 Stage 4: Categorical Encoding")

            if 'student_clean' not in st.session_state:
                st.warning("Please complete data cleaning first")
            else:
                df_clean = st.session_state.student_clean

                # Identify categorical columns
                categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                                   'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                                   'nursery', 'higher', 'internet', 'romantic']
                categorical_cols = [col for col in categorical_cols if col in df_clean.columns]

                if categorical_cols:
                    df_encoded = df_clean.copy()
                    encoding_map = {}

                    for col in categorical_cols:
                        le = LabelEncoder()
                        original_values = df_encoded[col].unique()
                        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                        encoded_values = df_encoded[col].unique()
                        encoding_map[col] = dict(zip(original_values, encoded_values))

                    # Store encoded data
                    st.session_state.student_encoded = df_encoded
                    st.session_state.student_encoding_map = encoding_map

                    # Show encoding mapping (first few columns)
                    st.subheader("Encoding Mapping (First 5 columns)")
                    for i, (col, mapping) in enumerate(list(encoding_map.items())[:5]):
                        st.write(f"**{col}:**")
                        for original, encoded in mapping.items():
                            st.write(f"  `{original}` → `{encoded}`")
                        if i < 4:
                            st.write("---")

                    if len(encoding_map) > 5:
                        st.info(f"... and {len(encoding_map) - 5} more categorical columns encoded")

                    st.dataframe(df_encoded.head(), use_container_width=True)
                    st.success(f"✅ {len(categorical_cols)} categorical variables encoded successfully")
                else:
                    st.session_state.student_encoded = df_clean
                    st.info("No categorical columns found to encode")

        # Stage 5: Normalization
        with tab4:
            st.markdown("### 📏 Stage 5: Feature Normalization")

            if 'student_encoded' not in st.session_state:
                st.warning("Please complete encoding first")
            else:
                df_encoded = st.session_state.student_encoded

                # Identify numerical columns (excluding target G3)
                numerical_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                                 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
                numerical_cols = [col for col in numerical_cols if col in df_encoded.columns and col != 'G3']

                if numerical_cols:
                    df_normalized = df_encoded.copy()

                    # Apply Standard Scaler
                    scaler = StandardScaler()
                    df_normalized[numerical_cols] = scaler.fit_transform(df_normalized[numerical_cols])

                    # Store normalized data and scaler
                    st.session_state.student_normalized = df_normalized
                    st.session_state.student_scaler = scaler

                    # Show comparison for key columns
                    st.subheader("Normalization Comparison (Key Features)")
                    key_cols = ['age', 'absences', 'G1', 'G2']  # Show only key columns for readability
                    key_cols = [col for col in key_cols if col in numerical_cols]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Before Normalization**")
                        st.dataframe(df_encoded[key_cols].describe())

                    with col2:
                        st.write("**After Normalization**")
                        st.dataframe(df_normalized[key_cols].describe())

                    st.success(f"✅ {len(numerical_cols)} numerical features normalized with Standard Scaler")
                else:
                    st.session_state.student_normalized = df_encoded
                    st.info("No numerical columns found to normalize")

        # Stage 6: Train/Test Split
        with tab5:
            st.markdown("### ✂️ Stage 6: Train/Test Split")

            if 'student_normalized' not in st.session_state:
                st.warning("Please complete normalization first")
            else:
                df_normalized = st.session_state.student_normalized

                # Prepare features and target
                X = df_normalized.drop('G3', axis=1)
                y = df_normalized['G3']

                # Split data (80% train, 20% test) - different ratio for this dataset
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Store split data
                st.session_state.student_X_train = X_train
                st.session_state.student_X_test = X_test
                st.session_state.student_y_train = y_train
                st.session_state.student_y_test = y_test

                # Show results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Training Set")
                    st.metric("X_train shape", f"{X_train.shape[0]} × {X_train.shape[1]}")
                    st.metric("y_train shape", f"{y_train.shape[0]}")

                    # Grade statistics in train
                    st.write("**Grade statistics (G3):**")
                    st.write(f"- Mean: {y_train.mean():.2f}")
                    st.write(f"- Min: {y_train.min()}")
                    st.write(f"- Max: {y_train.max()}")

                with col2:
                    st.subheader("Test Set")
                    st.metric("X_test shape", f"{X_test.shape[0]} × {X_test.shape[1]}")
                    st.metric("y_test shape", f"{y_test.shape[0]}")

                    # Grade statistics in test
                    st.write("**Grade statistics (G3):**")
                    st.write(f"- Mean: {y_test.mean():.2f}")
                    st.write(f"- Min: {y_test.min()}")
                    st.write(f"- Max: {y_test.max()}")

                st.success("✅ Data split completed (80% train, 20% test)")

        # Results Summary
        with tab6:
            st.markdown("### 📊 Preprocessing Results Summary")

            if all(key in st.session_state for key in ['student_X_train', 'student_X_test', 'student_y_train', 'student_y_test']):
                st.markdown('<div class="success-message">🎉 All preprocessing stages completed successfully!</div>', unsafe_allow_html=True)

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Original Rows", f"{st.session_state.student_raw.shape[0]:,}")
                with col2:
                    st.metric("Final Rows", f"{st.session_state.student_X_train.shape[0] + st.session_state.student_X_test.shape[0]:,}")
                with col3:
                    st.metric("Features", st.session_state.student_X_train.shape[1])
                with col4:
                    st.metric("Train/Test Ratio", "80/20")

                # Show final datasets
                st.subheader("Training Data Preview")
                train_preview = st.session_state.student_X_train.copy()
                train_preview['G3'] = st.session_state.student_y_train
                st.dataframe(train_preview.head(), use_container_width=True)

                st.subheader("Test Data Preview")
                test_preview = st.session_state.student_X_test.copy()
                test_preview['G3'] = st.session_state.student_y_test
                st.dataframe(test_preview.head(), use_container_width=True)

                # Export options
                st.subheader("💾 Export Processed Data")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("📊 Export Training Data", key="student_train_export"):
                        train_df = st.session_state.student_X_train.copy()
                        train_df['G3'] = st.session_state.student_y_train
                        csv = train_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Training CSV",
                            csv,
                            "student_performance_train_processed.csv",
                            "text/csv"
                        )

                with col2:
                    if st.button("📊 Export Test Data", key="student_test_export"):
                        test_df = st.session_state.student_X_test.copy()
                        test_df['G3'] = st.session_state.student_y_test
                        csv = test_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Test CSV",
                            csv,
                            "student_performance_test_processed.csv",
                            "text/csv"
                        )
            else:
                st.warning("Please complete all preprocessing stages to see the summary")

elif dataset_key == "iris":
    # Iris Dataset Page
    st.markdown('<div class="main-header">🌸 Iris Dataset - Species Classification</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 📋 Dataset Overview
    The Iris dataset is a classic multivariate dataset introduced by Ronald Fisher in 1936.
    It contains measurements of 150 iris flowers from three different species. This page provides
    a complete 6-stage preprocessing pipeline for species classification.
    """)

    # Stage 1: Data Loading
    st.markdown('<div class="section-header">📊 Stage 1: Data Loading</div>', unsafe_allow_html=True)

    if st.button("🔄 Load Iris Dataset", type="primary"):
        df = load_iris_data()

        if df is not None:
            st.session_state.iris_raw = df.copy()
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Flowers", f"{df.shape[0]:,}")
            with col2:
                species_count = df['species'].nunique()
                st.metric("Species Count", species_count)
            with col3:
                most_common = df['species'].mode()[0]
                st.metric("Most Common Species", most_common)

            st.dataframe(df.head(), use_container_width=True)
        else:
            st.error("❌ Failed to load dataset")

    # Check if data is loaded
    if 'iris_raw' not in st.session_state:
        st.info("👆 Please load the dataset first to continue with preprocessing.")
    else:
        df = st.session_state.iris_raw

        # Create tabs for preprocessing stages
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔍 Exploration",
            "🧹 Cleaning",
            "🔤 Encoding",
            "📏 Normalization",
            "✂️ Train/Test Split",
            "📊 Results Summary"
        ])

        # Stage 2: Data Exploration
        with tab1:
            st.markdown("### 🔍 Stage 2: Data Exploration")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Data Types")
                dtypes_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str)
                })
                st.dataframe(dtypes_df, use_container_width=True)

            with col2:
                st.subheader("Missing Values")
                null_counts = df.isnull().sum()
                if null_counts.sum() == 0:
                    st.success("✅ No missing values found")
                else:
                    null_df = pd.DataFrame({
                        'Column': null_counts.index,
                        'Missing': null_counts.values,
                        'Percentage': (null_counts / len(df) * 100).round(2)
                    })
                    st.dataframe(null_df[null_df['Missing'] > 0], use_container_width=True)

            # Basic statistics
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)

            # Species distribution and feature analysis
            st.subheader("Dataset Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Species distribution
                st.write("**Species Distribution**")
                species_counts = df['species'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                ax.bar(species_counts.index, species_counts.values, color=colors, alpha=0.7)
                ax.set_title('Distribution of Iris Species')
                ax.set_xlabel('Species')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot(fig)

            with col2:
                # Feature correlations
                st.write("**Feature Correlations**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                               square=True, ax=ax)
                    ax.set_title('Feature Correlation Matrix')
                    st.pyplot(fig)

            # Feature distributions by species
            st.subheader("Feature Distributions by Species")
            feature_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            species_list = df['species'].unique()

            for i, feature in enumerate(feature_cols):
                ax = axes[i]
                for j, species in enumerate(species_list):
                    species_data = df[df['species'] == species][feature]
                    ax.hist(species_data, alpha=0.7, label=species, color=colors[j], bins=15)
                ax.set_title(f'{feature}')
                ax.set_xlabel('Value (cm)')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

        # Stage 3: Data Cleaning
        with tab2:
            st.markdown("### 🧹 Stage 3: Data Cleaning")

            # For Iris dataset, minimal cleaning needed
            # No irrelevant columns to remove, no missing values
            df_clean = df.copy()

            # Remove duplicates if any
            initial_rows = df_clean.shape[0]
            df_clean = df_clean.drop_duplicates()
            final_rows = df_clean.shape[0]

            # Store cleaned data
            st.session_state.iris_clean = df_clean

            # Show results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Before Cleaning")
                st.metric("Rows", df.shape[0])
                st.metric("Columns", df.shape[1])
                st.metric("Missing Values", df.isnull().sum().sum())

            with col2:
                st.subheader("After Cleaning")
                st.metric("Rows", df_clean.shape[0])
                st.metric("Columns", df_clean.shape[1])
                st.metric("Missing Values", df_clean.isnull().sum().sum())

            if initial_rows > final_rows:
                st.success(f"✅ Removed {initial_rows - final_rows} duplicate rows")
            else:
                st.success("✅ Dataset was already clean - no duplicates or missing values")

            st.dataframe(df_clean.head(), use_container_width=True)

        # Stage 4: Encoding
        with tab3:
            st.markdown("### 🔤 Stage 4: Categorical Encoding")

            if 'iris_clean' not in st.session_state:
                st.warning("Please complete data cleaning first")
            else:
                df_clean = st.session_state.iris_clean

                # The Iris dataset has only one categorical column: 'species'
                if 'species' in df_clean.columns:
                    df_encoded = df_clean.copy()

                    # Label encode the species column
                    le = LabelEncoder()
                    original_values = df_encoded['species'].unique()
                    df_encoded['species'] = le.fit_transform(df_encoded['species'])
                    encoded_values = df_encoded['species'].unique()

                    encoding_map = {'species': dict(zip(original_values, encoded_values))}

                    # Store encoded data
                    st.session_state.iris_encoded = df_encoded
                    st.session_state.iris_encoding_map = encoding_map

                    # Show encoding mapping
                    st.subheader("Encoding Mapping")
                    st.write("**species (target variable):**")
                    for original, encoded in encoding_map['species'].items():
                        st.write(f"  `{original}` → `{encoded}`")

                    st.dataframe(df_encoded.head(), use_container_width=True)
                    st.success("✅ Species variable encoded successfully")
                else:
                    st.session_state.iris_encoded = df_clean
                    st.info("No categorical columns found to encode")

        # Stage 5: Normalization
        with tab4:
            st.markdown("### 📏 Stage 5: Feature Normalization")

            if 'iris_encoded' not in st.session_state:
                st.warning("Please complete encoding first")
            else:
                df_encoded = st.session_state.iris_encoded

                # Identify numerical columns (excluding target 'species')
                numerical_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
                numerical_cols = [col for col in numerical_cols if col in df_encoded.columns and col != 'species']

                if numerical_cols:
                    df_normalized = df_encoded.copy()

                    # Apply Standard Scaler
                    scaler = StandardScaler()
                    df_normalized[numerical_cols] = scaler.fit_transform(df_normalized[numerical_cols])

                    # Store normalized data and scaler
                    st.session_state.iris_normalized = df_normalized
                    st.session_state.iris_scaler = scaler

                    # Show comparison
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Before Normalization")
                        st.dataframe(df_encoded[numerical_cols].describe())

                    with col2:
                        st.subheader("After Normalization")
                        st.dataframe(df_normalized[numerical_cols].describe())

                    # Visualization of normalization effect
                    st.subheader("Normalization Effect Visualization")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                    # Before normalization
                    df_encoded[numerical_cols].boxplot(ax=ax1)
                    ax1.set_title('Before Normalization')
                    ax1.set_ylabel('Value')
                    ax1.tick_params(axis='x', rotation=45)

                    # After normalization
                    df_normalized[numerical_cols].boxplot(ax=ax2)
                    ax2.set_title('After Normalization (Standard Scaler)')
                    ax2.set_ylabel('Standardized Value')
                    ax2.tick_params(axis='x', rotation=45)

                    plt.tight_layout()
                    st.pyplot(fig)

                    st.success("✅ 4 numerical features normalized with Standard Scaler")
                else:
                    st.session_state.iris_normalized = df_encoded
                    st.info("No numerical columns found to normalize")

        # Stage 6: Train/Test Split
        with tab5:
            st.markdown("### ✂️ Stage 6: Train/Test Split")

            if 'iris_normalized' not in st.session_state:
                st.warning("Please complete normalization first")
            else:
                df_normalized = st.session_state.iris_normalized

                # Prepare features and target
                X = df_normalized.drop('species', axis=1)
                y = df_normalized['species']

                # Split data (70% train, 30% test)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )

                # Store split data
                st.session_state.iris_X_train = X_train
                st.session_state.iris_X_test = X_test
                st.session_state.iris_y_train = y_train
                st.session_state.iris_y_test = y_test

                # Show results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Training Set")
                    st.metric("X_train shape", f"{X_train.shape[0]} × {X_train.shape[1]}")
                    st.metric("y_train shape", f"{y_train.shape[0]}")

                    # Class distribution in train
                    train_species = y_train.value_counts().sort_index()
                    st.write("**Species distribution:**")
                    species_names = ['setosa', 'versicolor', 'virginica']
                    for i, count in enumerate(train_species):
                        st.write(f"- {species_names[i]}: {count} ({count/len(y_train)*100:.1f}%)")

                with col2:
                    st.subheader("Test Set")
                    st.metric("X_test shape", f"{X_test.shape[0]} × {X_test.shape[1]}")
                    st.metric("y_test shape", f"{y_test.shape[0]}")

                    # Class distribution in test
                    test_species = y_test.value_counts().sort_index()
                    st.write("**Species distribution:**")
                    for i, count in enumerate(test_species):
                        st.write(f"- {species_names[i]}: {count} ({count/len(y_test)*100:.1f}%)")

                st.success("✅ Data split completed with stratification")

        # Results Summary
        with tab6:
            st.markdown("### 📊 Preprocessing Results Summary")

            if all(key in st.session_state for key in ['iris_X_train', 'iris_X_test', 'iris_y_train', 'iris_y_test']):
                st.markdown('<div class="success-message">🎉 All preprocessing stages completed successfully!</div>', unsafe_allow_html=True)

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Original Rows", f"{st.session_state.iris_raw.shape[0]:,}")
                with col2:
                    st.metric("Final Rows", f"{st.session_state.iris_X_train.shape[0] + st.session_state.iris_X_test.shape[0]:,}")
                with col3:
                    st.metric("Features", st.session_state.iris_X_train.shape[1])
                with col4:
                    st.metric("Train/Test Ratio", "70/30")

                # Show final datasets
                st.subheader("Training Data Preview")
                train_preview = st.session_state.iris_X_train.copy()
                train_preview['species'] = st.session_state.iris_y_train
                # Convert back to species names for readability
                species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
                train_preview['species'] = train_preview['species'].map(species_map)
                st.dataframe(train_preview.head(), use_container_width=True)

                st.subheader("Test Data Preview")
                test_preview = st.session_state.iris_X_test.copy()
                test_preview['species'] = st.session_state.iris_y_test
                test_preview['species'] = test_preview['species'].map(species_map)
                st.dataframe(test_preview.head(), use_container_width=True)

                # Export options
                st.subheader("💾 Export Processed Data")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("📊 Export Training Data", key="iris_train_export"):
                        train_df = st.session_state.iris_X_train.copy()
                        train_df['species'] = st.session_state.iris_y_train
                        # Convert back to species names for export
                        train_df['species'] = train_df['species'].map(species_map)
                        csv = train_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Training CSV",
                            csv,
                            "iris_train_processed.csv",
                            "text/csv"
                        )

                with col2:
                    if st.button("📊 Export Test Data", key="iris_test_export"):
                        test_df = st.session_state.iris_X_test.copy()
                        test_df['species'] = st.session_state.iris_y_test
                        # Convert back to species names for export
                        test_df['species'] = test_df['species'].map(species_map)
                        csv = test_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Test CSV",
                            csv,
                            "iris_test_processed.csv",
                            "text/csv"
                        )
            else:
                st.warning("Please complete all preprocessing stages to see the summary")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 About")
st.sidebar.info("""
**ML Preprocessing Lab v2.0**

A comprehensive tool for machine learning data preprocessing with an intuitive interface.
""")

st.sidebar.markdown("### 🔗 Links")
st.sidebar.markdown("""
- [Streamlit Docs](https://docs.streamlit.io)
- [Scikit-learn](https://scikit-learn.org)
- [Pandas](https://pandas.pydata.org)
""")
