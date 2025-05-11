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
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        
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
        if df[feature].nunique() <= 2:
            sns.countplot(x=feature, hue='Diabetes_binary', data=df, ax=ax)
        else:
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
        
        # Create input fields for all features in the correct order
        inputs = {}
        cols = st.columns(3)  # Create 3 columns for better layout
        
        # Get feature ranges for sliders
        feature_ranges = {col: (df[col].min(), df[col].max()) for col in X.columns}
        
        # Organize features into the columns
        features_per_col = len(X.columns) // 3
        remaining = len(X.columns) % 3
        
        for i, col in enumerate(X.columns):
            # Determine which column to use
            col_idx = min(i // (features_per_col + (1 if i < remaining else 0)), 2)
            
            with cols[col_idx]:
                if df[col].nunique() == 2:  # Binary feature
                    inputs[col] = st.selectbox(col, [0, 1], key=col)
                else:
                    min_val, max_val = feature_ranges[col]
                    if col in ['BMI', 'Age']:  # Example of specific handling for certain features
                        step = 1.0 if col == 'BMI' else 1
                        inputs[col] = st.slider(col, float(min_val), float(max_val), 
                                              float(df[col].median()), step=step, key=col)
                    else:
                        inputs[col] = st.slider(col, int(min_val), int(max_val), 
                                              int(df[col].median()), key=col)
        
        # Create dataframe from inputs in the correct feature order
        input_df = pd.DataFrame([inputs])[X.columns]  # Ensure same column order as training data
        
        # Make prediction
        if st.button("Predict Diabetes Risk"):
            try:
                # Scale input
                input_scaled = scaler.transform(input_df)
                
                # Get prediction and probability
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
                
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
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    main()
