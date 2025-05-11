import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# Set plot style
sns.set(style="whitegrid")

# Load dataset function
@st.cache_data
def load_data():
    try:
        # Updated working URL
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
        df = pd.read_csv(url, header=None)
        
        # Add column names to match the expected format
        columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Diabetes_binary'
        ]
        df.columns = columns
        
        st.success("Dataset loaded successfully from GitHub!")
        return df
        
    except Exception as e:
        st.warning(f"Failed to load from URL: {str(e)}")
        st.info("Trying local file...")
        
        try:
            # For local development
            df = pd.read_csv("diabetes.csv")
            st.success("Dataset loaded successfully from local file!")
            return df
        except:
            st.error("""
            Failed to load data. Please ensure:
            1. For online use: Internet connection is available
            2. For local use: Place 'diabetes.csv' in same directory
            """)
            return None

def show_data_exploration(df):
    st.header("Data Exploration")
    
    st.subheader("Dataset Overview")
    st.write("First 5 rows of the dataset:")
    st.write(df.head())
    
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    
    st.subheader("Class Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Diabetes_binary', data=df, ax=ax)
    plt.title('Distribution of Diabetes')
    plt.xlabel('Diabetes (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    st.pyplot(fig)

def show_feature_analysis(df):
    st.header("Feature Analysis")
    
    # Select feature to analyze
    feature = st.selectbox("Select feature to analyze", df.columns[:-1])  # Exclude target
    
    # Plot distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Diabetes_binary', y=feature, data=df, ax=ax)
    plt.title(f'Distribution of {feature} by Diabetes Status')
    plt.xlabel('Diabetes (0 = No, 1 = Yes)')
    plt.ylabel(feature)
    st.pyplot(fig)
    
    # Correlation with diabetes
    corr = df[['Diabetes_binary', feature]].corr().iloc[0,1]
    st.write(f"Correlation with diabetes: {corr:.2f}")

def show_prediction(df):
    st.header("Diabetes Risk Prediction")
    
    # Prepare data for modeling
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Display metrics
    st.subheader("Model Performance")
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")
    
    st.write("\nClassification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.table(pd.DataFrame(report).transpose())
    
    st.write("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)
    
    # Feature importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance.head(10), ax=ax)
    plt.title('Top 10 Important Features')
    st.pyplot(fig)
    
    # Prediction form
    st.subheader("Make a Prediction")
    st.write("Enter health indicators to predict diabetes risk:")
    
    # Create input fields for top 5 features
    top_features = importance['Feature'].head(5).tolist()
    inputs = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        for feature in top_features[:3]:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            default_val = float(df[feature].median())
            inputs[feature] = st.slider(
                feature, 
                min_val, 
                max_val, 
                default_val,
                help=f"Range: {min_val}-{max_val}"
            )
    
    with col2:
        for feature in top_features[3:5]:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            default_val = float(df[feature].median())
            inputs[feature] = st.slider(
                feature, 
                min_val, 
                max_val, 
                default_val,
                help=f"Range: {min_val}-{max_val}"
            )
    
    # Set default values for other features
    default_values = {col: float(df[col].median()) for col in X.columns}
    input_data = {**default_values, **inputs}
    
    # Convert to DataFrame with same column order as training data
    input_df = pd.DataFrame([input_data])[X.columns]
    
    # Make prediction
    if st.button("Predict Diabetes Risk"):
        try:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            if prediction == 1:
                st.error(f"High risk of diabetes ({probability:.1%} probability)")
                st.info("Consult with a healthcare professional.")
            else:
                st.success(f"Low risk of diabetes ({probability:.1%} probability)")
                st.info("Maintain healthy lifestyle habits.")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

def main():
    st.title("Diabetes Prediction App")
    st.write("""
    This app analyzes health indicators to predict diabetes risk using the Pima Indians Diabetes Dataset.
    """)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.info("""
        Using sample data for demonstration. For full functionality:
        1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
        2. Save as 'diabetes.csv' in the same directory
        """)
        # Create sample data
        df = pd.DataFrame({
            'Pregnancies': [6, 1, 8],
            'Glucose': [148, 85, 183],
            'BloodPressure': [72, 66, 64],
            'SkinThickness': [35, 29, 0],
            'Insulin': [0, 0, 0],
            'BMI': [33.6, 26.6, 23.3],
            'DiabetesPedigreeFunction': [0.627, 0.351, 0.672],
            'Age': [50, 31, 32],
            'Diabetes_binary': [1, 0, 1]
        })
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Data Exploration", "Feature Analysis", "Prediction"])
    
    if page == "Data Exploration":
        show_data_exploration(df)
    elif page == "Feature Analysis":
        show_feature_analysis(df)
    elif page == "Prediction":
        show_prediction(df)

if __name__ == "__main__":
    main()
