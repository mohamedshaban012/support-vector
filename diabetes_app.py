import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page Configuration
st.set_page_config(
    page_title="BRFSS Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .st-bb { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; }
    .risk-high { color: #e74c3c; font-weight: bold; font-size: 1.3em; }
    .risk-moderate { color: #f39c12; font-weight: bold; font-size: 1.3em; }
    .risk-low { color: #2ecc71; font-weight: bold; font-size: 1.3em; }
</style>
""", unsafe_allow_html=True)

# Load Dataset
@st.cache_data
def load_data():
    """Load the BRFSS diabetes dataset from local path"""
    try:
        file_path = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
        if not os.path.exists(file_path):
            st.error("Dataset not found at specified path!")
            st.stop()
            
        df = pd.read_csv(file_path)
        
        # Verify required columns exist
        if 'Diabetes_binary' not in df.columns:
            st.error("Target column 'Diabetes_binary' not found in dataset")
            st.stop()
            
        return df
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

# Main App
def main():
    st.title("🩺 BRFSS Diabetes Risk Prediction")
    st.markdown("Using the 2015 Behavioral Risk Factor Surveillance System dataset")
    
    # Navigation
    app_mode = st.sidebar.radio("Select Mode", 
                               ["📊 Data Explorer", 
                                "⚙️ Model Training", 
                                "🔮 Predict Risk"])
    
    df = load_data()
    
    if app_mode == "📊 Data Explorer":
        st.header("Dataset Exploration")
        
        with st.expander("Dataset Summary", expanded=True):
            st.write(f"📐 Shape: {df.shape}")
            st.write("🔍 First 5 rows:")
            st.dataframe(df.head())
            
            st.write("📝 Feature Description:")
            desc = {
                'Diabetes_binary': '0=No diabetes, 1=Prediabetes/Diabetes',
                'HighBP': '0=No high BP, 1=High BP',
                'HighChol': '0=No high cholesterol, 1=High cholesterol',
                'BMI': 'Body Mass Index',
                'GenHlth': '1-5 scale (1=Excellent, 5=Poor)'
            }
            st.table(pd.DataFrame.from_dict(desc, orient='index', columns=['Description']))
        
        # Visualizations
        st.subheader("Data Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            feature = st.selectbox("Select Feature", df.columns.drop('Diabetes_binary'))
            fig = px.histogram(df, x=feature, color='Diabetes_binary', 
                              barmode='overlay', title=f"{feature} Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("Correlation with Diabetes")
            corr = df.corr()[['Diabetes_binary']].sort_values('Diabetes_binary', ascending=False)
            st.dataframe(corr.style.background_gradient(cmap='Blues'))
    
    elif app_mode == "⚙️ Model Training":
        st.header("SVM Model Training")
        
        # Data Preparation
        X = df.drop('Diabetes_binary', axis=1)
        y = df['Diabetes_binary']
        
        # Feature Selection
        st.subheader("Feature Selection")
        k = st.slider("Number of Features to Select", 5, 20, 10)
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
        st.write("Selected Features:")
        st.write(selected_features.tolist())
        
        # Model Training
        st.subheader("Model Training")
        if st.button("Train SVM Model"):
            with st.spinner('Training... This may take a few minutes'):
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=5)),
                    ('svm', SVC(kernel='rbf', probability=True, cache_size=1000))
                ])
                
                model.fit(X[selected_features], y)
                joblib.dump(model, 'brfss_svm_model.joblib')
                st.success("Model trained and saved successfully!")
                
                # Quick evaluation
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, X[selected_features], y, cv=3)
                st.write(f"Cross-validation Accuracy: {np.mean(scores):.2%}")
    
    else:  # Prediction Mode
        st.header("Diabetes Risk Prediction")
        
        if not os.path.exists('brfss_svm_model.joblib'):
            st.warning("Please train the model first in the Model Training section")
            st.stop()
            
        model = joblib.load('brfss_svm_model.joblib')
        
        # Input Form
        st.subheader("Patient Information")
        col1, col2 = st.columns(2)
        
        inputs = {}
        with col1:
            inputs['HighBP'] = st.selectbox("High Blood Pressure", [0, 1])
            inputs['HighChol'] = st.selectbox("High Cholesterol", [0, 1])
            inputs['BMI'] = st.slider("BMI", 10.0, 50.0, 25.0)
            
        with col2:
            inputs['GenHlth'] = st.slider("General Health (1-5)", 1, 5, 3)
            inputs['Age'] = st.slider("Age Group", 1, 13, 5)
            inputs['PhysHlth'] = st.slider("Physical Health Days", 0, 30, 0)
        
        if st.button("Calculate Diabetes Risk"):
            input_df = pd.DataFrame([inputs])
            proba = model.predict_proba(input_df)[0][1]
            
            # Display Results
            st.subheader("Results")
            st.metric("Diabetes Risk Probability", f"{proba:.1%}")
            
            # Visual Risk Indicator
            if proba > 0.7:
                st.markdown('<p class="risk-high">🛑 High Risk of Diabetes</p>', unsafe_allow_html=True)
            elif proba > 0.4:
                st.markdown('<p class="risk-moderate">⚠️ Moderate Risk of Diabetes</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="risk-low">✅ Low Risk of Diabetes</p>', unsafe_allow_html=True)
            
            # Explanation
            with st.expander("Risk Factors Analysis"):
                st.write("""
                - **High Blood Pressure**: Increases risk by 30-50%
                - **BMI > 30**: Major risk factor
                - **Poor General Health**: Strong correlation
                - **Age**: Risk increases after 45
                """)

if __name__ == "__main__":
    main()
