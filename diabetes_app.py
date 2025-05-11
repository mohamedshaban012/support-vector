import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io

# Set page config
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

# Set plot style
sns.set(style="whitegrid")

# Load dataset function
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    return df

def create_input_widget(col, series):
    """Helper function to create appropriate input widget based on data type"""
    if series.nunique() <= 2:  # Binary feature
        return st.selectbox(col, sorted(series.unique()))
    else:  # Numeric feature
        min_val = float(series.min())
        max_val = float(series.max())
        value = float(series.median())
        
        # Handle case where min and max are equal
        if min_val == max_val:
            st.write(f"{col}: {value} (fixed value)")
            return value
        
        step = 1.0 if series.dtype == 'float64' else 1.0
        return st.slider(col, min_val, max_val, value, step=step)

def main():
    st.title("Diabetes Prediction App (Top 5 Features)")
    st.write("""
    This app analyzes the most important health indicators to predict diabetes risk.
    The model uses only the top 5 most predictive features from the BRFSS 2015 survey.
    """)
    
    # Load data
    df = load_data()
    
    # Prepare data for modeling
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model to get feature importance
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Get top 5 features
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    top_5_features = importance['Feature'].head(5).tolist()
    X_top5 = X[top_5_features]
    
    # Retrain model with top 5 features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_top5)
    X_test_scaled = scaler.transform(X_test[top_5_features])
    
    model_top5 = RandomForestClassifier(random_state=42)
    model_top5.fit(X_train_scaled, y_train)
    
    # Sidebar options
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Feature Importance", "Model Performance", "Prediction"])
    
    if page == "Feature Importance":
        st.header("Top 5 Most Important Features")
        
        # Show feature importance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance.head(5), ax=ax)
        plt.title('Top 5 Important Features for Diabetes Prediction')
        st.pyplot(fig)
        
        # Show description of each feature
        st.subheader("Feature Descriptions")
        feature_descriptions = {
            "HighBP": "Have you ever been told you have high blood pressure? (0=No, 1=Yes)",
            "HighChol": "Have you ever been told you have high cholesterol? (0=No, 1=Yes)",
            "BMI": "Body Mass Index (weight in kg/(height in m)^2)",
            "GenHlth": "Would you say your general health is: (1=Excellent, 5=Poor)",
            "Age": "Age in years (13-level category)"
        }
        
        for feature in top_5_features:
            st.write(f"**{feature}**: {feature_descriptions.get(feature, 'No description available')}")
    
    elif page == "Model Performance":
        st.header("Model Performance with Top 5 Features")
        
        # Make predictions
        y_pred = model_top5.predict(X_test_scaled)
        
        # Display metrics
        st.subheader("Accuracy Score")
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
    
    elif page == "Prediction":
        st.header("Diabetes Risk Prediction")
        st.write("Enter values for the top 5 most important health indicators:")
        
        # Create input fields for top 5 features
        inputs = {}
        cols = st.columns(2)  # Create 2 columns for better layout
        
        # First column
        with cols[0]:
            for feature in top_5_features[:3]:
                inputs[feature] = create_input_widget(feature, df[feature])
        
        # Second column
        with cols[1]:
            for feature in top_5_features[3:]:
                inputs[feature] = create_input_widget(feature, df[feature])
        
        # Make prediction
        if st.button("Predict Diabetes Risk"):
            try:
                # Create dataframe from inputs in the correct feature order
                input_df = pd.DataFrame([inputs])[top_5_features]
                
                # Scale input
                input_scaled = scaler.transform(input_df)
                
                # Get prediction and probability
                prediction = model_top5.predict(input_scaled)[0]
                probability = model_top5.predict_proba(input_scaled)[0][1]
                
                # Display results
                st.subheader("Prediction Results")
                if prediction == 1:
                    st.error(f"High risk of diabetes (probability: {probability:.2%})")
                else:
                    st.success(f"Low risk of diabetes (probability: {probability:.2%})")
                
                # Show interpretation
                st.write("""
                **Interpretation:**
                - Probability < 30%: Low risk
                - Probability 30-70%: Moderate risk
                - Probability > 70%: High risk
                """)
                
                # Show feature values used
                st.write("\n**Input Values Used:**")
                for feature, value in inputs.items():
                    st.write(f"- {feature}: {value}")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    main()
