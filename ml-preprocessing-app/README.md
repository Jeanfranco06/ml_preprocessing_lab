# 🤖 ML Data Preprocessing Suite

A comprehensive, interactive Streamlit application for machine learning data preprocessing across multiple datasets. This application demonstrates professional data preprocessing techniques with beautiful visualizations and export capabilities.

## 🎯 Features

### 📊 **Multi-Dataset Support**
- **Titanic**: Survival prediction preprocessing
- **Student Performance**: Grade prediction analysis
- **Iris**: Species classification preprocessing

### 🔄 **Complete Preprocessing Pipeline**
1. **Initial Exploration**: Data types, missing values, descriptive statistics
2. **Data Cleaning**: Handle nulls, remove duplicates, detect outliers
3. **Encoding**: Label encoding and one-hot encoding for categorical variables
4. **Normalization**: Standard scaling and min-max scaling options
5. **Train/Test Split**: Configurable data splitting with stratification
6. **Advanced Visualizations**: Dataset-specific insights and correlations

### 🎨 **Professional UI/UX**
- **Interactive Tabs**: Step-by-step preprocessing workflow
- **Real-time Metrics**: Live updates of data transformations
- **Beautiful Visualizations**: Plotly and Seaborn charts
- **Responsive Design**: Works on desktop and mobile
- **Dark/Light Theme**: User preference selection

### 💾 **Export Capabilities**
- **Multiple Formats**: CSV, Excel, JSON export
- **Processing Summary**: Detailed transformation logs
- **Pipeline Code Generation**: Automated Python code export
- **Data Download**: Processed datasets at any stage

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd ml-preprocessing-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place datasets in the `datasets/` folder**
   - `titanic.csv` - Titanic survival dataset
   - `student-mat.csv` - Student performance dataset
   - Iris dataset loads automatically from scikit-learn

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## 📁 Project Structure

```
ml-preprocessing-app/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── datasets/                   # Dataset storage
│   ├── titanic.csv            # Titanic dataset (user provided)
│   └── student-mat.csv        # Student dataset (user provided)
├── utils/                      # Utility modules
│   ├── preprocessing.py       # Data preprocessing functions
│   ├── visualization.py       # Chart and visualization functions
│   └── export.py              # Export and download functions
└── pages/                      # Individual dataset pages
    ├── titanic.py             # Titanic-specific analysis
    ├── student.py             # Student performance analysis
    └── iris.py                # Iris dataset analysis
```

## 🎓 Educational Value

This application serves as a comprehensive learning tool for:

- **Data Preprocessing Techniques**: Complete ML pipeline understanding
- **Interactive Data Science**: Hands-on experience with real datasets
- **Visualization Best Practices**: Professional chart creation
- **Streamlit Development**: Web app creation for data science
- **Export and Deployment**: Making data science results shareable

## 📊 Dataset Details

### Titanic Dataset
- **Source**: Kaggle Titanic competition
- **Objective**: Predict passenger survival
- **Features**: Demographics, ticket info, cabin details
- **Preprocessing**: Remove irrelevant columns, handle missing ages/embarked, encode categories

### Student Performance Dataset
- **Source**: Kaggle Student Alcohol Consumption
- **Objective**: Predict final grades (G3)
- **Features**: Demographics, family background, study habits
- **Preprocessing**: One-hot encoding, handle categorical variables, normalize grades

### Iris Dataset
- **Source**: scikit-learn built-in dataset
- **Objective**: Species classification
- **Features**: Sepal/petal measurements
- **Preprocessing**: Standardization, minimal cleaning needed

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Export**: openpyxl, fpdf2
- **UI Components**: streamlit-extras

## 📈 Usage Examples

### Basic Workflow
1. Select a dataset from the sidebar
2. Explore initial data statistics and visualizations
3. Clean data (handle missing values, remove duplicates)
4. Encode categorical variables
5. Normalize numerical features
6. Split into train/test sets
7. View advanced visualizations and insights
8. Export processed data and generated code

### Advanced Features
- **Interactive Controls**: Adjust preprocessing parameters
- **Method Comparison**: Compare different normalization techniques
- **Progress Tracking**: Real-time processing status
- **Code Generation**: Export complete preprocessing pipelines

## 🤝 Contributing

This is an educational project. Feel free to:
- Add more datasets
- Implement additional preprocessing techniques
- Improve visualizations
- Add more export formats
- Enhance the UI/UX

## 📄 License

This project is for educational purposes. Datasets are from public sources with appropriate licenses.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web app framework
- **scikit-learn** for machine learning utilities
- **Kaggle** for datasets and community
- **Plotly** and **Seaborn** for visualization libraries

---

**Happy Learning! 🚀**

*Built with ❤️ for machine learning education*
