import pandas as pd
import streamlit as st
from io import BytesIO
import base64

def show_export_options(original_df, cleaned_df, encoded_df, normalized_df, dataset_key):
    """Display export options for processed data"""

    st.subheader("Export Processed Data")

    # Export format selection
    export_format = st.selectbox(
        "Select export format",
        ["CSV", "Excel", "JSON"],
        help="Choose the format for exporting your processed data"
    )

    # Data stage selection
    data_options = {
        "Original Data": original_df,
        "Cleaned Data": cleaned_df,
        "Encoded Data": encoded_df,
        "Normalized Data": normalized_df
    }

    # Filter out None values
    available_data = {k: v for k, v in data_options.items() if v is not None}

    selected_stage = st.selectbox(
        "Select data processing stage to export",
        list(available_data.keys())
    )

    df_to_export = available_data[selected_stage]

    if df_to_export is not None:
        # Show preview
        st.subheader(f"Preview of {selected_stage}")
        st.dataframe(df_to_export.head())

        # Export button
        if st.button(f"📥 Download {selected_stage} as {export_format}"):
            export_data(df_to_export, export_format, f"{dataset_key}_{selected_stage.lower().replace(' ', '_')}")

        # Generate processing summary
        st.subheader("📋 Processing Summary")
        show_processing_summary(original_df, cleaned_df, encoded_df, normalized_df, dataset_key)

        # Export pipeline code
        st.subheader("🔧 Export Processing Pipeline")
        if st.button("💾 Generate Pipeline Code"):
            pipeline_code = generate_pipeline_code(dataset_key)
            st.code(pipeline_code, language="python")

            # Download pipeline code
            st.download_button(
                "📥 Download Pipeline Code",
                pipeline_code,
                file_name=f"{dataset_key}_preprocessing_pipeline.py",
                mime="text/plain"
            )

def export_data(df, format_type, filename):
    """Export dataframe in specified format"""
    if format_type == "CSV":
        csv_data = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv_data,
            file_name=f"{filename}.csv",
            mime="text/csv"
        )

    elif format_type == "Excel":
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Processed_Data', index=False)
        buffer.seek(0)
        st.download_button(
            "Download Excel",
            buffer,
            file_name=f"{filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    elif format_type == "JSON":
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            "Download JSON",
            json_data,
            file_name=f"{filename}.json",
            mime="application/json"
        )

def show_processing_summary(original_df, cleaned_df, encoded_df, normalized_df, dataset_key):
    """Show summary of processing steps applied"""

    summary_data = {
        "Stage": [],
        "Rows": [],
        "Columns": [],
        "Changes": []
    }

    # Original
    summary_data["Stage"].append("Original")
    summary_data["Rows"].append(original_df.shape[0])
    summary_data["Columns"].append(original_df.shape[1])
    summary_data["Changes"].append("Raw data")

    # Cleaned
    if cleaned_df is not None:
        summary_data["Stage"].append("Cleaned")
        summary_data["Rows"].append(cleaned_df.shape[0])
        summary_data["Columns"].append(cleaned_df.shape[1])

        changes = []
        if cleaned_df.shape[0] != original_df.shape[0]:
            changes.append(f"Removed {original_df.shape[0] - cleaned_df.shape[0]} rows")
        if cleaned_df.shape[1] != original_df.shape[1]:
            changes.append(f"Removed {original_df.shape[1] - cleaned_df.shape[1]} columns")
        if original_df.isnull().sum().sum() > cleaned_df.isnull().sum().sum():
            null_diff = original_df.isnull().sum().sum() - cleaned_df.isnull().sum().sum()
            changes.append(f"Handled {null_diff} null values")

        summary_data["Changes"].append("; ".join(changes) if changes else "No changes")

    # Encoded
    if encoded_df is not None:
        summary_data["Stage"].append("Encoded")
        summary_data["Rows"].append(encoded_df.shape[0])
        summary_data["Columns"].append(encoded_df.shape[1])

        prev_df = cleaned_df if cleaned_df is not None else original_df
        changes = []
        if encoded_df.shape[1] != prev_df.shape[1]:
            changes.append(f"Added {encoded_df.shape[1] - prev_df.shape[1]} encoded columns")

        summary_data["Changes"].append("; ".join(changes) if changes else "Categorical encoding applied")

    # Normalized
    if normalized_df is not None:
        summary_data["Stage"].append("Normalized")
        summary_data["Rows"].append(normalized_df.shape[0])
        summary_data["Columns"].append(normalized_df.shape[1])
        summary_data["Changes"].append("Numerical features scaled")

    # Display summary table
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

def generate_pipeline_code(dataset_key):
    """Generate Python code for the preprocessing pipeline"""

    if dataset_key == "titanic":
        code = '''
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_titanic_data(df):
    """
    Complete preprocessing pipeline for Titanic dataset
    """
    # 1. Remove irrelevant columns
    df_clean = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, errors='ignore')

    # 2. Handle missing values
    df_clean['Age'].fillna(df_clean['Age'].mean(), inplace=True)
    df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)

    # 3. Remove duplicates
    df_clean = df_clean.drop_duplicates()

    # 4. Encode categorical variables
    le = LabelEncoder()
    df_clean['Sex'] = le.fit_transform(df_clean['Sex'])
    df_clean['Embarked'] = le.fit_transform(df_clean['Embarked'])

    # 5. Normalize numerical features
    scaler = StandardScaler()
    numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
    existing_num_cols = [col for col in numerical_cols if col in df_clean.columns]
    df_clean[existing_num_cols] = scaler.fit_transform(df_clean[existing_num_cols])

    # 6. Split data
    X = df_clean.drop('Survived', axis=1)
    y = df_clean['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, scaler

# Usage example:
# df = pd.read_csv('titanic.csv')
# X_train, X_test, y_train, y_test, scaler = preprocess_titanic_data(df)
'''
    elif dataset_key == "student_performance":
        code = '''
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_student_data(df):
    """
    Complete preprocessing pipeline for Student Performance dataset
    """
    # 1. Remove duplicates
    df_clean = df.drop_duplicates()

    # 2. Handle missing values (if any)
    df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    df_clean = df_clean.fillna(df_clean.mode().iloc[0])

    # 3. One-hot encode categorical variables
    categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                       'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                       'nursery', 'higher', 'internet', 'romantic']

    existing_cat_cols = [col for col in categorical_cols if col in df_clean.columns]
    df_encoded = pd.get_dummies(df_clean, columns=existing_cat_cols, drop_first=False)

    # 4. Normalize numerical features
    scaler = MinMaxScaler()
    numerical_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                     'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']

    existing_num_cols = [col for col in numerical_cols if col in df_encoded.columns]
    df_encoded[existing_num_cols] = scaler.fit_transform(df_encoded[existing_num_cols])

    # 5. Split data
    X = df_encoded.drop('G3', axis=1)
    y = df_encoded['G3']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler

# Usage example:
# df = pd.read_csv('student-mat.csv')
# X_train, X_test, y_train, y_test, scaler = preprocess_student_data(df)
'''
    elif dataset_key == "iris":
        code = '''
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_iris_data():
    """
    Complete preprocessing pipeline for Iris dataset
    """
    # 1. Load data
    iris = load_iris()
    df = pd.DataFrame(
        data=np.c_[iris['data'], iris['target']],
        columns=iris['feature_names'] + ['target']
    )

    # 2. Standardize features
    scaler = StandardScaler()
    X = df.drop('target', axis=1)
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=iris['feature_names'])
    df_scaled['target'] = df['target']

    # 3. Split data
    X = df_scaled.drop('target', axis=1)
    y = df_scaled['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, scaler

# Usage example:
# X_train, X_test, y_train, y_test, scaler = preprocess_iris_data()
'''
    else:
        code = "# Pipeline code generation not available for this dataset"

    return code
