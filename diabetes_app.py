import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

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
        # Load the provided dataset
        df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
        
        # No need to convert since it's already binary
        st.success("Dataset loaded successfully!")
        return df
        
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None

def show_data_exploration(df):
    st.header("Data Exploration")
    
    st.subheader("Dataset Overview")
    st.write(f"Number of records: {df.shape[0]:,}")
    st.write(f"Number of features: {df.shape[1]}")
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
    feature = st.selectbox("Select feature to analyze", df.columns[:-1])
    
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

def train_and_evaluate_model(df):
    st.header("Model Training and Evaluation")
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Prepare data
    X = df_imputed.drop('Diabetes_binary', axis=1)
    y = df_imputed['Diabetes_binary']
    
    # Feature selection
    selector = SelectKBest(f_classif, k=15)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    
    # Dimensionality reduction
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_selected)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42))
    ])
    
    # Hyperparameter tuning
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['linear', 'rbf'],
        'svm__gamma': ['scale', 'auto']
    }
    
    with st.spinner('Training SVM with Grid Search...'):
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
    
    st.success("Model training completed!")
    st.write("Best Parameters:", grid_search.best_params_)
    
    # Evaluate model
    y_pred = grid_search.predict(X_test)
    y_proba = grid_search.predict_proba(X_test)[:, 1]
    
    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    st.write("\nClassification Report:")
    st.text(classification_report(y_test, y_pred))
    
    st.write("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)
    
    return grid_search.best_estimator_, selector, pca, selected_features

def show_prediction(model, selector, pca, selected_features, df):
    st.header("Diabetes Risk Prediction")
    
    st.subheader("Make a Prediction")
    st.write("Enter health indicators to predict diabetes risk:")
    
    # Create default values dictionary
    default_values = {
        'HighBP': 0,
        'HighChol': 0,
        'CholCheck': 1,
        'BMI': 25,
        'Smoker': 0,
        'Stroke': 0,
        'HeartDiseaseorAttack': 0,
        'PhysActivity': 1,
        'Fruits': 1,
        'Veggies': 1,
        'HvyAlcoholConsump': 0,
        'AnyHealthcare': 1,
        'NoDocbcCost': 0,
        'GenHlth': 3,
        'MentHlth': 0,
        'PhysHlth': 0,
        'DiffWalk': 0,
        'Sex': 0,
        'Age': 5,
        'Education': 4,
        'Income': 6
    }
    
    # Create input fields for top features
    inputs = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        for feature in selected_features[:len(selected_features)//2]:
            if df[feature].nunique() == 2:  # Binary feature
                inputs[feature] = st.selectbox(feature, [0, 1], index=default_values.get(feature, 0))
            else:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                default_val = float(default_values.get(feature, (min_val + max_val)/2))
                inputs[feature] = st.slider(feature, min_val, max_val, default_val)
    
    with col2:
        for feature in selected_features[len(selected_features)//2:]:
            if df[feature].nunique() == 2:  # Binary feature
                inputs[feature] = st.selectbox(feature, [0, 1], index=default_values.get(feature, 0))
            else:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                default_val = float(default_values.get(feature, (min_val + max_val)/2))
                inputs[feature] = st.slider(feature, min_val, max_val, default_val)
    
    # Create input dataframe with ALL features in correct order
    input_data = {}
    for feature in df.columns:
        if feature == 'Diabetes_binary':
            continue
        if feature in inputs:  # If user provided this feature
            input_data[feature] = inputs[feature]
        else:  # Set default value for other features
            input_data[feature] = default_values.get(feature, 0)
    
    # Convert to DataFrame with same column order as original data
    input_df = pd.DataFrame([input_data])[df.drop('Diabetes_binary', axis=1).columns]
    
    # Make prediction
    if st.button("Predict Diabetes Risk"):
        try:
            # Apply feature selection and PCA
            input_selected = selector.transform(input_df)
            input_pca = pca.transform(input_selected)
            
            # Scale and predict
            input_scaled = StandardScaler().fit_transform(input_pca)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            if prediction == 1:
                st.error(f"High risk of diabetes (probability: {probability:.2%})")
                st.info("Recommendation: Consult with a healthcare professional.")
            else:
                st.success(f"Low risk of diabetes (probability: {probability:.2%})")
                st.info("Recommendation: Maintain healthy lifestyle habits.")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

def main():
    st.title("Diabetes Prediction App")
    st.write("""
    This app analyzes health indicators to predict diabetes risk using SVM.
    The dataset is a balanced 50-50 split of diabetes cases from BRFSS 2015 survey.
    """)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", [
        "Data Exploration", 
        "Feature Analysis", 
        "Model Training",
        "Prediction"
    ])
    
    if page == "Data Exploration":
        show_data_exploration(df)
    elif page == "Feature Analysis":
        show_feature_analysis(df)
    elif page == "Model Training":
        if 'model' not in st.session_state:
            model, selector, pca, selected_features = train_and_evaluate_model(df)
            st.session_state['model'] = model
            st.session_state['selector'] = selector
            st.session_state['pca'] = pca
            st.session_state['selected_features'] = selected_features
            st.session_state['df'] = df  # Store df in session state
        else:
            st.info("Model already trained. Go to Prediction page.")
    elif page == "Prediction":
        if 'model' not in st.session_state:
            st.warning("Please train the model first on the Model Training page")
        else:
            show_prediction(
                st.session_state['model'],
                st.session_state['selector'],
                st.session_state['pca'],
                st.session_state['selected_features'],
                st.session_state['df']  # Pass df from session state
            )

if __name__ == "__main__":
    main()
