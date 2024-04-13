import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

class DiabetesPredictionApp:
    def __init__(self):
        self.df = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.rf_clf = None
        self.et_clf = None
        self.lgb_clf = None
        self.lr_clf = None
        self.ensemble_clf_rf_et = None

    @staticmethod
    @st.cache_data
    def load_data():
        return pd.read_csv('diabetes_prediction_dataset.csv')

    def run(self):
        self.df = self.load_data()
        self.sidebar()
        self.main_content()

    def sidebar(self):
        st.sidebar.title("Options")
        self.view_data = st.sidebar.checkbox("View Dataset")
        self.show_summary = st.sidebar.checkbox("Show Summary Statistics")
        self.show_visualizations = st.sidebar.checkbox("Show Visualizations")
        self.show_preprocessing = st.sidebar.checkbox("Show Data Preprocessing")
        self.show_model_training = st.sidebar.checkbox("Train Models")
        self.show_evaluation_metrics = st.sidebar.checkbox("Show Evaluation Metrics")
        self.make_predictions = st.sidebar.checkbox("Make Predictions")

    def main_content(self):
        st.title("Diabetes Prediction Streamlit App")

        if self.view_data:
            self.view_dataset()

        if self.show_summary:
            self.show_summary_statistics()

        if self.show_visualizations:
            self.show_data_visualizations()

        if self.show_preprocessing:
            self.preprocess_data()

        if self.show_model_training:
            if self.X_train_scaled is not None and self.y_train is not None:
                self.train_models()
            else:
                st.warning("Please preprocess the data first.")

        if self.show_evaluation_metrics:
            if self.X_test_scaled is not None and self.y_test is not None:
                self.evaluate_models()
            else:
                st.warning("Please preprocess the data and train the models first.")

        if self.make_predictions:
            if self.X_test_scaled is not None:
                self.predict()
            else:
                st.warning("Please preprocess the data and train the models first.")

    def view_dataset(self):
        st.subheader("Dataset")
        st.write(self.df)

    def show_summary_statistics(self):
        st.subheader("Summary Statistics")
        st.write(self.df.describe())

    def show_data_visualizations(self):
        # Bar charts for variables not indicated in the box plot
        boxplot_vars = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        barplot_vars = [col for col in self.df.columns if col not in boxplot_vars]
        for var in barplot_vars:
            st.write(f"Bar Chart of {var}")
            plt.figure(figsize=(8, 6))
            sns.countplot(x=var, data=self.df)
            plt.title(f"Bar Chart of {var}")
            plt.xlabel(var)
            plt.ylabel("Count")
            st.pyplot()

        # Histograms for variables indicated in the box plot
        st.write("Histograms of variables indicated in the Boxplot")
        for var in boxplot_vars:
            st.write(f"Histogram of {var}")
            plt.figure(figsize=(8, 6))
            plt.hist(self.df[var], bins=20, color='skyblue', edgecolor='black')
            plt.title(f'Histogram of {var}')
            plt.xlabel(var)
            plt.ylabel('Frequency')
            plt.grid(True)
            st.pyplot()

        # Multiple box plots for selected variables
        st.write("Multiple Box Plots")
        selected_vars = st.multiselect("Select variables for Box Plots", self.df.columns)
        if selected_vars:
            plt.figure(figsize=(15, 8))
            sns.boxplot(data=self.df[selected_vars])
            plt.title("Multiple Box Plots")
            plt.xlabel("Variables")
            plt.ylabel("Values")
            plt.xticks(rotation=45)
            st.pyplot()

    def preprocess_data(self):
        # Add debugging statements to check data preprocessing steps
        st.write("Debugging: Checking data preprocessing steps...")
        st.write("Original DataFrame:")
        st.write(self.df.head())
    
        # Checking missing values
        missing_values = self.df.isna().sum()
        st.write("Missing Values:")
        st.write(missing_values)
    
        # Handling missing values
        replace_missing = st.selectbox("Replace missing values with:", ['Mean', 'Median', 'Mode', 'Remove'])
        if replace_missing != 'Remove':
            for column in self.df.columns:
                if self.df[column].dtype == 'object':
                    mode_val = self.df[column].mode()[0]
                    self.df.fillna({column: mode_val}, inplace=True)
                else:
                    if replace_missing == 'Mean':
                        mean_val = self.df[column].mean()
                        self.df[column] = self.df[column].fillna(mean_val)
                    elif replace_missing == 'Median':
                        median_val = self.df[column].median()
                        self.df[column] = self.df[column].fillna(median_val)
                    else:  # Mode
                        mode_val = self.df[column].mode()[0]
                        self.df.fillna({column: mode_val}, inplace=True)
        else:
            self.df.dropna(inplace=True)
    
        # Outlier treatment
        outlier_action = st.selectbox("Outlier Treatment:", ['Remove', 'Replace with Mean', 'Replace with Median', 'Replace with Lower/Upper Quantile'])
        if outlier_action != 'Remove':
            for column in self.boxplot_vars:
                if outlier_action == 'Replace with Mean':
                    mean_val = self.df[column].mean()
                    self.df[column] = np.where((self.df[column] < self.df[column].quantile(0.25)) | (self.df[column] > self.df[column].quantile(0.75)), mean_val, self.df[column])
                elif outlier_action == 'Replace with Median':
                    median_val = self.df[column].median()
                    self.df[column] = np.where((self.df[column] < self.df[column].quantile(0.25)) | (self.df[column] > self.df[column].quantile(0.75)), median_val, self.df[column])
                else:  # Replace with Lower/Upper Quantile
                    lower_quantile = self.df[column].quantile(0.25)
                    upper_quantile = self.df[column].quantile(0.75)
                    self.df[column] = np.where(self.df[column] < lower_quantile, lower_quantile, self.df[column])
                    self.df[column] = np.where(self.df[column] > upper_quantile, upper_quantile, self.df[column])
    
        # Label encoding for categorical variables
        label_encoder = LabelEncoder()
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                self.df[column] = label_encoder.fit_transform(self.df[column])
    
        # Balancing the response variable using SMOTE
        sm = SMOTE(sampling_strategy='minority', random_state=42)
        X_resampled, y_resampled = sm.fit_resample(self.df.drop(['diabetes'], axis=1), self.df['diabetes'])
        df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    
        # Train-test split
        X = df_resampled.drop(['diabetes'], axis=1)
        y = df_resampled['diabetes']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Feature scaling
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Check attribute assignment after train-test split
        st.write("Debugging: Checking attribute assignment after train-test split...")
        st.write("X_train_scaled:", self.X_train_scaled.shape)
        st.write("y_train:", self.y_train.shape)
        st.write("X_test_scaled:", self.X_test_scaled.shape)
        st.write("y_test:", self.y_test.shape)
    
        st.write("Data Preprocessing Completed!")
        pass
    
    def train_models(self):
        
        # Debugging: Checking attribute assignment before fitting the models...
        st.write("Debugging: Checking attribute assignment before fitting the models...")
        st.write("X_train_scaled:", self.X_train_scaled)
        st.write("y_train:", self.y_train)
        st.write("X_test_scaled:", self.X_test_scaled)
        st.write("y_test:", self.y_test)
    
        # Model training
        params_rf = {'n_estimators': 100, 'random_state': 42}
        params_et = {'n_estimators': 100, 'random_state': 42}
        params_lgb = {'n_estimators': 100, 'random_state': 42}
        params_lr = {'random_state': 42}
    
        self.rf_clf = RandomForestClassifier(**params_rf)
        self.et_clf = ExtraTreesClassifier(**params_et)
        self.lgb_clf = lgb.LGBMClassifier(**params_lgb)
        self.lr_clf = LogisticRegression(**params_lr)
    
        # Fit the models
        self.rf_clf.fit(self.X_train_scaled, self.y_train)
        self.et_clf.fit(self.X_train_scaled, self.y_train)
        self.lgb_clf.fit(self.X_train_scaled, self.y_train)
        self.lr_clf.fit(self.X_train_scaled, self.y_train)
    
        # Create the ensemble classifier
        self.ensemble_clf_rf_et = VotingClassifier(estimators=[('rf', self.rf_clf), ('et', self.et_clf)], voting='soft')
    
        # Fit the ensemble classifier
        self.ensemble_clf_rf_et.fit(self.X_train_scaled, self.y_train)
    
        st.write("Models Trained Successfully!")
        pass

    def evaluate_models(self):
        st.subheader("Model Evaluation")
        selected_model = st.selectbox("Select Model for Evaluation", ["Random Forest", "Extra Trees", "LightGBM", "Logistic Regression", "Ensemble (RF + ET)"])
        if selected_model == "Random Forest":
            model = self.rf_clf
        elif selected_model == "Extra Trees":
            model = self.et_clf
        elif selected_model == "LightGBM":
            model = self.lgb_clf
        elif selected_model == "Logistic Regression":
            model = self.lr_clf
        elif selected_model == "Ensemble (RF + ET)":
            model = self.ensemble_clf_rf_et

        if model:
            scores = cross_validate(model, self.X_test_scaled, self.y_test, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'])
            st.write(f"Model: {selected_model}")
            st.write("Accuracy:", np.mean(scores['test_accuracy']))
            st.write("Precision:", np.mean(scores['test_precision']))
            st.write("Recall:", np.mean(scores['test_recall']))
            st.write("F1 Score:", np.mean(scores['test_f1']))
            st.write("Confusion Matrix:")
            y_pred = model.predict(self.X_test_scaled)
            cm = confusion_matrix(self.y_test, y_pred)
            st.write(cm)
        else:
            st.warning("Please train the models first.")

    def make_predictions(self):
        st.subheader("Make Predictions")
        st.write("Please input values for prediction:")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, step=0.1)
        hbA1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, step=0.1)
        blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0.0, max_value=600.0, step=1.0)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        hypertension = st.selectbox("Hypertension", ['Yes', 'No'])
        heart_disease = st.selectbox("Heart Disease", ['Yes', 'No'])
        smoking_history = st.selectbox("Smoking History", ['Yes', 'No'])
    
        gender = 1 if gender == 'Male' else 0
        hypertension = 1 if hypertension == 'Yes' else 0
        heart_disease = 1 if heart_disease == 'Yes' else 0
        smoking_history = 1 if smoking_history == 'Yes' else 0
    
        input_data = np.array([[age, bmi, hbA1c_level, blood_glucose_level, gender, hypertension, heart_disease, smoking_history]])
        input_data_scaled = self.scaler.transform(input_data)
    
        st.write("Predictions:")
        st.write("Random Forest Classifier:", self.rf_clf.predict(input_data_scaled))
        st.write("Extra Trees Classifier:", self.et_clf.predict(input_data_scaled))
        st.write("LightGBM Classifier:", self.lgb_clf.predict(input_data_scaled))
        st.write("Logistic Regression:", self.lr_clf.predict(input_data_scaled))
        st.write("Combine Model of Random Forest and Extra Trees:", self.ensemble_clf_rf_et.predict(input_data_scaled))
        
        # Adding the target variable
        st.write("Expected Outcome:")
        expected_outcome = "Diabetes" if input("Do you have diabetes? ('Yes' or 'No') ").strip().lower() == 'yes' else "No Diabetes"
        st.write(expected_outcome)
        pass


if __name__ == "__main__":
    app = DiabetesPredictionApp()
    app.run()
