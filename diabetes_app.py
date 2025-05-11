import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore

# Set page config
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

# Set plot style
sns.set(style="whitegrid")

# Load dataset function
@st.cache_data
def load_data():
    df = pd.read_csv("E:\downloads\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    return df

# Main function
def main():
    st.title("Diabetes Prediction App")
    st.write("""
    This app analyzes health indicators to predict diabetes risk.
    The dataset is from the BRFSS 2015 survey with a 50-50 split of diabetes cases.
    """)
    
    # Load data
    df = load_data()
    
    # Sidebar options
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Data Exploration", "Feature Analysis", "Prediction"])
    
    if page == "Data Exploration":
        st.header("Data Exploration")
        
        st.subheader("Dataset Overview")
        st.write("First 5 rows of the dataset:")
        st.write(df.head())
        
        st.write("\nDataset Info:")
        st.write(df.info())
        
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
        
    elif page == "Feature Analysis":
        st.header("Feature Analysis")
        
        # Select feature to analyze
        feature = st.selectbox("Select feature to analyze", df.columns[1:])
        
        # Plot distribution by diabetes status
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Diabetes_binary', y=feature, data=df, ax=ax)
        plt.title(f'Distribution of {feature} by Diabetes Status')
        plt.xlabel('Diabetes (0 = No, 1 = Yes)')
        plt.ylabel(feature)
        st.pyplot(fig)
        
        # Correlation with diabetes
        corr = df[['Diabetes_binary', feature]].corr().iloc[0,1]
        st.write(f"Correlation with diabetes: {corr:.2f}")
        
    elif page == "Prediction":
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
                if df[feature].nunique() == 2:  # Binary feature
                    inputs[feature] = st.selectbox(feature, [0, 1])
                else:
                    min_val = int(df[feature].min())
                    max_val = int(df[feature].max())
                    inputs[feature] = st.slider(feature, min_val, max_val)
        
        with col2:
            for feature in top_features[3:5]:
                if df[feature].nunique() == 2:  # Binary feature
                    inputs[feature] = st.selectbox(feature, [0, 1])
                else:
                    min_val = int(df[feature].min())
                    max_val = int(df[feature].max())
                    inputs[feature] = st.slider(feature, min_val, max_val)
        
        # Add remaining features with default values
        for feature in X.columns:
            if feature not in inputs:
                inputs[feature] = 0  # Default value
        
        # Create dataframe from inputs
        input_df = pd.DataFrame([inputs])
        
        # Make prediction
        if st.button("Predict Diabetes Risk"):
            # Scale input
            input_scaled = scaler.transform(input_df)
            
            # Get prediction and probability
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            if prediction == 1:
                st.error(f"High risk of diabetes (probability: {probability:.2%})")
            else:
                st.success(f"Low risk of diabetes (probability: {probability:.2%})")

if __name__ == "__main__":
    main()