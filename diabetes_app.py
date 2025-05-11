import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# Set page config
st.set_page_config(page_title="Diabetes Prediction with SVM", page_icon="🩺", layout="wide")

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
    st.title("Diabetes Prediction with SVM")
    st.write("""
    This app uses Support Vector Machine (SVM) to predict diabetes risk based on health indicators 
    from the BRFSS 2015 survey, following a complete ML pipeline with dimensionality reduction.
    """)
    
    # Load data
    try:
        df = load_data()
        
        # Show dataset info in expander
        with st.expander("Dataset Overview"):
            st.write(f"Shape: {df.shape}")
            st.write("First 5 rows:")
            st.write(df.head())
            st.write("Feature descriptions:")
            feature_descriptions = {
                "Diabetes_binary": "0 = no diabetes, 1 = prediabetes or diabetes",
                "HighBP": "0 = no high BP, 1 = high BP",
                "HighChol": "0 = no high cholesterol, 1 = high cholesterol",
                "BMI": "Body Mass Index",
                "Smoker": "Have you smoked at least 100 cigarettes in your entire life?",
                "Stroke": "Ever had a stroke",
                "HeartDiseaseorAttack": "Coronary heart disease or myocardial infarction",
                "PhysActivity": "Physical activity in past 30 days",
                "Fruits": "Consume fruit 1 or more times per day",
                "Veggies": "Consume vegetables 1 or more times per day",
                "HvyAlcoholConsump": "Heavy drinkers (adult men >=14 drinks/week and adult women>=7 drinks/week)",
                "AnyHealthcare": "Have any kind of health care coverage",
                "NoDocbcCost": "Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?",
                "GenHlth": "Would you say your general health is: scale 1-5 (1=excellent, 5=poor)",
                "MentHlth": "Days of poor mental health scale 1-30 days",
                "PhysHlth": "Physical illness or injury days in past 30 days",
                "DiffWalk": "Do you have serious difficulty walking or climbing stairs?",
                "Sex": "0 = female, 1 = male",
                "Age": "13-level age category",
                "Education": "Education level (1-6 scale)",
                "Income": "Income scale (1-8)"
            }
            st.table(pd.DataFrame.from_dict(feature_descriptions, orient='index').rename(columns={0: 'Description'}))
        
        # Data Cleaning Section
        with st.expander("Data Cleaning Report"):
            st.subheader("Missing Values")
            st.write("Number of missing values per feature:")
            st.write(df.isnull().sum())
            
            st.subheader("Outlier Analysis")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            st.write("Summary statistics for numeric features:")
            st.write(df[numeric_cols].describe())
        
        # Prepare data for modeling
        X = df.drop('Diabetes_binary', axis=1)
        y = df['Diabetes_binary']
        
        # Split data FIRST to prevent data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # EDA Section
        with st.expander("Exploratory Data Analysis"):
            st.subheader("Target Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x=y_train, ax=ax)
            ax.set_title("Diabetes Distribution (0=No, 1=Yes)")
            st.pyplot(fig)
            
            st.subheader("Feature Distributions")
            selected_feature = st.selectbox("Select feature to visualize", X.columns)
            fig, ax = plt.subplots()
            if X[selected_feature].nunique() <= 10:
                sns.countplot(x=X_train[selected_feature], ax=ax)
            else:
                sns.histplot(X_train[selected_feature], ax=ax, kde=True)
            ax.set_title(f"Distribution of {selected_feature}")
            st.pyplot(fig)
            
            st.subheader("Correlation Matrix")
            corr_matrix = X_train.corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        
        # Dimensionality Reduction
        with st.expander("Dimensionality Reduction"):
            st.subheader("Feature Selection")
            
            # Method selection
            reduction_method = st.radio(
                "Select dimensionality reduction method:",
                ("PCA", "SelectKBest")
            )
            
            if reduction_method == "PCA":
                n_components = st.slider("Number of PCA components", 1, X.shape[1], 5)
                pca = PCA(n_components=n_components)
                X_train_reduced = pca.fit_transform(X_train)
                X_test_reduced = pca.transform(X_test)
                
                # Plot explained variance
                fig, ax = plt.subplots()
                ax.bar(range(1, n_components+1), pca.explained_variance_ratio_)
                ax.set_xlabel("Principal Component")
                ax.set_ylabel("Variance Explained")
                ax.set_title("PCA Explained Variance")
                st.pyplot(fig)
                
                st.write(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
                
            else:  # SelectKBest
                k = st.slider("Number of features to select", 1, X.shape[1], 5)
                selector = SelectKBest(score_func=f_classif, k=k)
                X_train_reduced = selector.fit_transform(X_train, y_train)
                X_test_reduced = selector.transform(X_test)
                
                # Get selected features
                selected_features = X.columns[selector.get_support()]
                st.write("Selected features:", list(selected_features))
                
                # Plot feature scores
                fig, ax = plt.subplots()
                scores = selector.scores_[selector.get_support()]
                sns.barplot(x=selected_features, y=scores, ax=ax)
                ax.set_title("Feature Importance Scores")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
        
        # Model Development
        st.header("SVM Model Development")
        
        # Create pipeline with scaling and SVM
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, random_state=42))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': [1, 0.1, 0.01, 0.001],
            'svm__kernel': ['rbf', 'linear']
        }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, n_jobs=-1, verbose=1
        )
        
        with st.spinner("Training SVM with hyperparameter tuning..."):
            grid_search.fit(X_train_reduced, y_train)
        
        best_model = grid_search.best_estimator_
        
        st.subheader("Model Evaluation")
        st.write(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test_reduced)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.write(f"Accuracy: {accuracy:.2%}")
        
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.table(pd.DataFrame(report).transpose())
        
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
        # Prediction Interface
        st.header("Diabetes Risk Prediction")
        st.write("Enter values for health indicators:")
        
        # Create input fields for all features
        inputs = {}
        cols = st.columns(2)  # Create 2 columns for better layout
        
        # First column
        with cols[0]:
            for feature in X.columns[:len(X.columns)//2]:
                inputs[feature] = create_input_widget(feature, df[feature])
        
        # Second column
        with cols[1]:
            for feature in X.columns[len(X.columns)//2:]:
                inputs[feature] = create_input_widget(feature, df[feature])
        
        # Make prediction
        if st.button("Predict Diabetes Risk"):
            try:
                # Create dataframe from inputs
                input_df = pd.DataFrame([inputs])
                
                # Apply same dimensionality reduction as training
                if reduction_method == "PCA":
                    input_reduced = pca.transform(input_df)
                else:
                    input_reduced = selector.transform(input_df)
                
                # Get prediction and probability
                prediction = best_model.predict(input_reduced)[0]
                probability = best_model.predict_proba(input_reduced)[0][1]
                
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
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your dataset and try again.")

if __name__ == "__main__":
    main()
