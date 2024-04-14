import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # Importing train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('diabetes_prediction_dataset.csv')

# Data preprocessing
def preprocess_data(df):
    # Drop missing values
    df.dropna(inplace=True)
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    # Remove outliers
    numeric_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Model training
def train_model(df):        
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()    
    # Apply label encoding to each categorical column
    for column in df.columns:
        if df[column].dtype == 'object':  # Check if column is categorical
            df[column] = label_encoder.fit_transform(df[column])
            
    X = df.drop(['diabetes'], axis=1)
    y = df['diabetes']
    # Feature selection
    selected_features = ['age', 'smoking_history','bmi', 'HbA1c_level', 'blood_glucose_level']
    X_selected = X[selected_features]
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    # Balancing the dataset
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    # Train the models
    params_rf = {
        'bootstrap': True, 
        'ccp_alpha': 0.0, 
        'class_weight': None,
        'criterion': 'gini', 
        'max_depth': None, 
        'max_features': 'sqrt',
        'max_leaf_nodes': None, 
        'max_samples': None,
        'min_impurity_decrease': 0.0, 
        'min_samples_leaf': 1,
        'min_samples_split': 2, 
        'min_weight_fraction_leaf': 0.0,
        'n_estimators': 100, 
        'n_jobs': -1, 
        'oob_score': False,
        'random_state': 1646, 
        'verbose': 0, 
        'warm_start': False
    }
    params_et = {
        'bootstrap': False,
        'ccp_alpha': 0.0,
        'class_weight': None,
        'criterion': 'gini',
        'max_depth': None,
        'max_features': 'sqrt',
        'max_leaf_nodes': None,
        'max_samples': None,
        'min_impurity_decrease': 0.0,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'min_weight_fraction_leaf': 0.0,
        'n_estimators': 100,
        'n_jobs': -1,
        'oob_score': False,
        'random_state': 8448,
        'verbose': 0,
        'warm_start': False
    }
    rf_model = RandomForestClassifier(**params_rf)
    et_model = ExtraTreesClassifier(**params_et)
    ensemble_rf_et = VotingClassifier(estimators=[('rf', rf_model), ('et', et_model)], voting='hard')
    ensemble_rf_et.fit(X_train_scaled, y_train_res)
    return ensemble_rf_et, X_test_scaled, y_test, scaler, selected_features

# Prediction
def predict_diabetes(model, input_data, scaler):
    input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    prediction = model.predict(input_data_scaled)
    return prediction

# Sidebar
st.sidebar.title('Navigation')
section = st.sidebar.radio('Go to', ('View Dataset', 'Data Summary', 'Data Visualization', 'Data Preprocessing', 'Model Training', 'Make Predictions'))

# Main content
st.title('Diabetes Prediction App')

if section == 'View Dataset':
    st.header('Dataset')
    df = load_data()
    st.write(df)

elif section == 'Data Summary':
    st.header('Data Summary')
    df = load_data()
    st.subheader('Numeric Columns Summary')
    st.write(df.describe())
    st.subheader('Categorical Columns Summary')
    categorical_columns = [column for column in df.columns if column not in ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
    for column in categorical_columns:
        st.write(f'Value counts for column: {column}')
        st.write(df[column].value_counts())

elif section == 'Data Visualization':
    st.header('Data Visualization')
    df = load_data()
    numeric_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    selected_variable = st.selectbox('Select a variable', df.columns)
    
    if selected_variable:
        if selected_variable in numeric_columns:
            if st.checkbox('Histogram'):
                plt.figure(figsize=(8, 6))
                plt.hist(df[selected_variable], bins=20, color='skyblue', edgecolor='black')
                plt.title(f'Histogram of {selected_variable}')
                plt.xlabel(selected_variable)
                plt.ylabel('Frequency')
                plt.grid(True)
                st.pyplot()
            if st.checkbox('Boxplot'):
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=df[selected_variable])
                plt.title(f'Boxplot of {selected_variable}')
                plt.grid(True)
                st.pyplot()
        else:
            if st.checkbox('Bar Plot'):
                plt.figure(figsize=(8, 6))
                sns.countplot(data=df, x=selected_variable)
                plt.title(f'Bar Plot of {selected_variable}')
                plt.xlabel(selected_variable)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot()


elif section == 'Data Preprocessing':
    st.header('Data Preprocessing')
    df = load_data()
    
    # Count missing values before preprocessing
    missing_values_before = df.isnull().sum().sum()
    
    # Count duplicates before dropping
    duplicate_rows_before = df[df.duplicated()].shape[0]
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Count outliers before preprocessing
    outliers_before = {}
    for column in ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']:
        Q1 = np.percentile(df[column], 25)
        Q3 = np.percentile(df[column], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        num_outliers = np.sum((df[column] < lower_bound) | (df[column] > upper_bound))
        outliers_before[column] = num_outliers
    
    # Remove outliers for all variables
    for column in ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']:
        Q1 = np.percentile(df[column], 25)
        Q3 = np.percentile(df[column], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Count missing values after preprocessing
    missing_values_after = df.isnull().sum().sum()
    
    # Count duplicates after dropping
    duplicate_rows_after = df[df.duplicated()].shape[0]
    
    # Count outliers after preprocessing
    outliers_after = {}
    for column in ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']:
        Q1 = np.percentile(df[column], 25)
        Q3 = np.percentile(df[column], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        num_outliers = np.sum((df[column] < lower_bound) | (df[column] > upper_bound))
        outliers_after[column] = num_outliers
    
    st.write('Data after preprocessing:')
    st.write(df.head())
    st.write('Number of missing values before preprocessing:', missing_values_before)
    st.write('Number of missing values after preprocessing:', missing_values_after)
    st.write('Number of duplicates before preprocessing:', duplicate_rows_before)
    st.write('Number of duplicates after preprocessing:', duplicate_rows_after)


elif section == 'Model Training':
    st.header('Model Training')
    ensemble_rf_et, X_test_scaled, y_test, scaler, selected_features = train_model(load_data())
    st.success('Models trained successfully!')

    # Model metrics
    y_pred = ensemble_rf_et.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    st.write('Model Metrics:')
    st.write(f'Accuracy: {accuracy:.2f}')
    st.write(f'Precision: {precision:.2f}')
    st.write(f'Recall: {recall:.2f}')
    st.write(f'F1 Score: {f1:.2f}')
    st.write(f'AUC: {auc:.2f}')

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    st.pyplot()

elif section == 'Make Predictions':
    st.header('Make Predictions')
    input_data = {}
    for feature in selected_features:
        input_data[feature] = st.number_input(f'Enter {feature}', step=0.01)
    if st.button('Predict'):
        prediction_ensemble = predict_diabetes(ensemble_rf_et, list(input_data.values()), scaler)
        st.write(f'Prediction using Ensemble Model (Random Forest + Extra Trees): {prediction_ensemble[0]}')
