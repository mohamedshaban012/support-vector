import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, roc_curve, auc)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="SVM Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# Set plot style
sns.set(style="whitegrid")

# Load dataset function
@st.cache_data
def load_data():
    try:
        # Using a large diabetes dataset (130,000+ records)
        url = "https://raw.githubusercontent.com/sonali-rai/SVM-Diabetes-Prediction/main/diabetes_012_health_indicators_BRFSS2015.csv"
        df = pd.read_csv(url)
        
        # Rename target column
        df = df.rename(columns={'Diabetes_012': 'Diabetes_binary'})
        
        # Convert to binary classification (0 = no diabetes, 1 = prediabetes or diabetes)
        df['Diabetes_binary'] = df['Diabetes_binary'].apply(lambda x: 0 if x == 0 else 1)
        
        st.success("Dataset loaded successfully!")
        return df
        
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None

def show_data_overview(df):
    st.header("1. Dataset Overview")
    
    st.subheader("Basic Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Number of records: {df.shape[0]:,}")
        st.write(f"Number of features: {df.shape[1]}")
    with col2:
        st.write(f"Target distribution:\n{df['Diabetes_binary'].value_counts(normalize=True)}")
    
    st.subheader("First 5 Rows")
    st.write(df.head())
    
    st.subheader("Feature Descriptions")
    feature_desc = {
        'Diabetes_binary': 'Target (0 = No diabetes, 1 = Prediabetes or Diabetes)',
        'HighBP': 'High blood pressure',
        'HighChol': 'High cholesterol',
        'CholCheck': 'Cholesterol check in past 5 years',
        'BMI': 'Body Mass Index',
        'Smoker': 'Have smoked at least 100 cigarettes',
        'Stroke': 'Ever had a stroke',
        'HeartDiseaseorAttack': 'Coronary heart disease or heart attack',
        'PhysActivity': 'Physical activity in past 30 days',
        'Fruits': 'Consume fruit daily',
        'Veggies': 'Consume vegetables daily',
        'HvyAlcoholConsump': 'Heavy alcohol consumption',
        'AnyHealthcare': 'Have any healthcare coverage',
        'NoDocbcCost': 'Could not see doctor due to cost',
        'GenHlth': 'General health (1=excellent to 5=poor)',
        'MentHlth': 'Days of poor mental health (past 30 days)',
        'PhysHlth': 'Days of poor physical health (past 30 days)',
        'DiffWalk': 'Difficulty walking or climbing stairs',
        'Sex': '0=Female, 1=Male',
        'Age': 'Age category (1-13)',
        'Education': 'Education level (1-6)',
        'Income': 'Income category (1-8)'
    }
    st.table(pd.DataFrame.from_dict(feature_desc, orient='index').rename(columns={0: 'Description'}))

def show_eda(df):
    st.header("2. Exploratory Data Analysis")
    
    st.subheader("Missing Values Analysis")
    st.write(df.isnull().sum())
    
    st.subheader("Correlation Matrix (Top 10 Features)")
    corr_matrix = df.corr().abs()
    top_features = corr_matrix['Diabetes_binary'].sort_values(ascending=False).index[:10]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select feature to visualize", df.columns[:-1])
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x=feature, color='Diabetes_binary', 
                          title=f'Distribution of {feature}')
        st.plotly_chart(fig1)
    with col2:
        fig2 = px.box(df, x='Diabetes_binary', y=feature, 
                     title=f'{feature} by Diabetes Status')
        st.plotly_chart(fig2)

def preprocess_data(df):
    st.header("3. Data Preprocessing")
    
    # Handle missing values
    st.subheader("Missing Values Treatment")
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Feature selection
    st.subheader("Feature Selection")
    X = df_imputed.drop('Diabetes_binary', axis=1)
    y = df_imputed['Diabetes_binary']
    
    selector = SelectKBest(f_classif, k=15)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    st.write("Selected Features:", list(selected_features))
    
    # Dimensionality reduction
    st.subheader("Dimensionality Reduction with PCA")
    pca = PCA(n_components=0.95)  # Retain 95% variance
    X_pca = pca.fit_transform(X_selected)
    st.write(f"Reduced from {X_selected.shape[1]} to {X_pca.shape[1]} components")
    
    # Plot explained variance
    fig = px.bar(x=range(1, pca.n_components_+1), 
                y=pca.explained_variance_ratio_,
                labels={'x': 'Principal Component', 'y': 'Variance Explained'},
                title='PCA Explained Variance')
    st.plotly_chart(fig)
    
    return X_pca, y, selected_features

def train_svm(X, y):
    st.header("4. SVM Model Development")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42))
    ])
    
    # Hyperparameter tuning
    st.subheader("Hyperparameter Tuning")
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['linear', 'rbf'],
        'svm__gamma': ['scale', 'auto']
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    with st.spinner('Training SVM with Grid Search...'):
        grid_search.fit(X_train, y_train)
    
    st.write("Best Parameters:", grid_search.best_params_)
    
    # Evaluate model
    st.subheader("Model Evaluation")
    y_pred = grid_search.predict(X_test)
    y_proba = grid_search.predict_proba(X_test)[:, 1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
    with col2:
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC = {roc_auc:.2f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'))
    fig.add_shape(type='line', line=dict(dash='dash'),
                 x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(fig)
    
    return grid_search.best_estimator_

def prediction_interface(model, selected_features):
    st.header("5. Prediction Interface")
    
    st.subheader("Make a Prediction")
    st.write("Enter values for the following features:")
    
    inputs = {}
    cols = st.columns(3)
    
    feature_ranges = {
        'HighBP': (0, 1),
        'HighChol': (0, 1),
        'BMI': (12, 98),
        'HeartDiseaseorAttack': (0, 1),
        'GenHlth': (1, 5),
        'PhysHlth': (0, 30),
        'DiffWalk': (0, 1),
        'Age': (1, 13),
        'Education': (1, 6),
        'Income': (1, 8)
    }
    
    for i, feature in enumerate(selected_features[:10]):  # Show first 10 for space
        with cols[i % 3]:
            if feature in feature_ranges:
                min_val, max_val = feature_ranges[feature]
                inputs[feature] = st.slider(
                    feature, min_val, max_val, (min_val + max_val) // 2)
            else:
                inputs[feature] = st.number_input(feature, value=0)
    
    if st.button("Predict Diabetes Risk"):
        try:
            # Create input array
            input_data = np.zeros(len(selected_features))
            for i, feat in enumerate(selected_features):
                input_data[i] = inputs.get(feat, 0)
            
            # Scale and predict
            prediction = model.predict([input_data])[0]
            probability = model.predict_proba([input_data])[0][1]
            
            if prediction == 1:
                st.error(f"High risk of diabetes ({probability:.1%} probability)")
                st.info("Recommendation: Consult with a healthcare professional.")
            else:
                st.success(f"Low risk of diabetes ({probability:.1%} probability)")
                st.info("Recommendation: Maintain healthy lifestyle habits.")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

def main():
    st.title("SVM Diabetes Prediction System")
    st.write("""
    A complete pipeline for diabetes prediction using Support Vector Machines (SVM)
    """)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", [
        "1. Dataset Overview",
        "2. Exploratory Data Analysis",
        "3. Data Preprocessing",
        "4. SVM Model Development",
        "5. Prediction Interface"
    ])
    
    if page == "1. Dataset Overview":
        show_data_overview(df)
    elif page == "2. Exploratory Data Analysis":
        show_eda(df)
    elif page == "3. Data Preprocessing":
        X, y, selected_features = preprocess_data(df)
        st.session_state['X'] = X
        st.session_state['y'] = y
        st.session_state['selected_features'] = selected_features
    elif page == "4. SVM Model Development":
        if 'X' not in st.session_state:
            st.warning("Please complete Data Preprocessing first")
        else:
            model = train_svm(st.session_state['X'], st.session_state['y'])
            st.session_state['model'] = model
    elif page == "5. Prediction Interface":
        if 'model' not in st.session_state:
            st.warning("Please train the model first")
        else:
            prediction_interface(
                st.session_state['model'],
                st.session_state['selected_features']
            )

if __name__ == "__main__":
    main()
