### Diabetes Prediction Streamlit App

This Streamlit app allows users to predict the likelihood of diabetes based on various health parameters. It includes the following features:

- **View Dataset:** Allows users to view the dataset used for training the models.
- **Show Summary Statistics:** Displays summary statistics of the dataset.
- **Show Visualizations:** Provides various visualizations such as bar charts, histograms, and box plots for data exploration.
- **Show Data Preprocessing:** Demonstrates data preprocessing steps including handling missing values, outlier treatment, and feature scaling.
- **Train Models:** Trains machine learning models including RandomForestClassifier, ExtraTreesClassifier, LogisticRegression, and an ensemble model.
- **Show Evaluation Metrics:** Displays evaluation metrics including accuracy, precision, recall, and confusion matrices for each model.
- **Make Predictions:** Allows users to input their health parameters and get predictions from trained models.

### How to Run

1. Clone the repository:

git clone https://github.com/DikoBrainiac/Prediction-of-Diabetes.git

2. Install dependencies:

pip install -r requirements.txt

3. Run the Streamlit app:

streamlit run Diabetes_Streamlit.py

### Dataset

The dataset used in this project is stored in `diabetes_prediction_dataset.csv`.

### Dependencies

The following Python libraries are required to run the app:

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- lightgbm
- imbalanced-learn

### Author

[DikoBrainiac](https://github.com/DikoBrainiac)


I added a comment