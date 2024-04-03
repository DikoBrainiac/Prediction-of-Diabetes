# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('diabetes_prediction_dataset.csv')

# %% [code]
df.head()

# %% [code]
df.info()

# %% [code] {"scrolled":true}
df

# %% [code]
df.shape

# %% [code]
#To describe the numeric columns in the data
df.describe()

# %% [code]
# To select the categorical variables from others
categorical_columns = ['gender','hypertension', 'heart_disease', 'smoking_history', 'diabetes']
for column in categorical_columns:
    # Count the occurrences of each unique value in the column
    value_counts = df[column].value_counts()
    
    # Print the value counts for the column
    print("Value counts for the column '{}'".format(column))
    print(value_counts)

# %% [code]
df.isna().sum()

# %% [code]
# checking for outliers
outliers = df[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
plt.figure(figsize=(15,14))
sns.boxplot(data=outliers)
plt.show()

# %% [code]
# Checking the number of outliers in bmi
# Calculate the first quartile (Q1)
Q1 = np.percentile(df['bmi'], 25)

# Calculate the third quartile (Q3)
Q3 = np.percentile(df['bmi'], 75)

# Calculate the interquartile range (IQR)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers detection
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Count the number of outliers
num_outliers = np.sum((df['bmi'] < lower_bound) | (df['bmi'] > upper_bound))

print("Number of outliers in BMI variable:", num_outliers)

# %% [code]
# Create a histogram of the BMI data
plt.figure(figsize=(8, 6))  # Set the figure size
plt.hist(df['bmi'], bins=20, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
plt.title('Histogram of BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.grid(True)  # Add grid lines for better readability
plt.show()

# %% [code]
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
for column in df.columns:
    if df[column].dtype == 'object':  # Check if column is categorical
        df[column] = label_encoder.fit_transform(df[column])

# %% [code]
df.head()

# %% [code]
X = df.drop(['diabetes'], axis=1)
y = df['diabetes']

# %% [code]
from sklearn.model_selection import train_test_split

# First, split into train+validation and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [code]
X_train.shape, y_train.shape

# %% [code]
y_train.value_counts()

# %% [code]
# Balancing the response variable
from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy='minority', random_state=42)
oversampled_X, oversampled_Y = sm.fit_resample(X_train, y_train)
oversampled = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)

# %% [code]
oversampled["diabetes"].value_counts()

# %% [code]
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
scaled_features = scaler.fit_transform(oversampled_X)

# Apply the same transformation to the validation and test sets
# scaled_vfeatures = scaler.transform(X_validation)
scaled_tfeatures = scaler.transform(X_test)

# Convert scaled_features back to DataFrame
X_train_scaled = pd.DataFrame(scaled_features, columns=X.columns)
# X_validation_scaled = pd.DataFrame(scaled_vfeatures, columns=X.columns)
X_test_scaled = pd.DataFrame(scaled_tfeatures, columns=X.columns)

scaled_df = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(scaled_features)], axis=1)

# %% [code]
X_train_scaled.head()

scaled_df.head()

# %% [code]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_validate

scoring = ['accuracy','precision', 'recall', 'f1', "roc_auc"]

params = {
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

# Create an Random forest classifier
rf_clf = RandomForestClassifier(**params) 

# Perform cross-validation with RFC
rf_clf_score = cross_validate(rf_clf, X=X_train_scaled, y=oversampled_Y, cv=10, scoring=scoring, return_train_score=True)
print(rf_clf_score)

# Compute the mean scores
mean_scores = {metric: np.mean(rf_clf_score[f'test_{metric}']) for metric in scoring}

print("Mean Scores:")
for metric, score in mean_scores.items():
    print(f"{metric}: {score:.4f}")

# %% [code]
import matplotlib.pyplot as plt

# Fit the Random Forest classifier to the training data
rf_clf.fit(X_train_scaled, oversampled_Y)

# Get feature importances
importances = rf_clf.feature_importances_

# Get feature names
feature_names = X_train_scaled.columns

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X_train_scaled.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_scaled.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X_train_scaled.shape[1]])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# %% [code]
from sklearn.ensemble import ExtraTreesClassifier

scoring = ['accuracy','precision', 'recall', 'f1', "roc_auc"]

params ={'bootstrap': False,
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
 'warm_start': False}
# Create an Extra Trees classifier
extra_trees_model = ExtraTreesClassifier(**params) 

# Perform cross-validation with Extra Trees classifier
extra_trees_score = cross_validate(extra_trees_model, X=X_train_scaled, y=oversampled_Y, cv=10, scoring=scoring, return_train_score=True)
print(extra_trees_score)

# Compute the mean scores
mean_scores = {metric: np.mean(extra_trees_score[f'test_{metric}']) for metric in scoring}

print("Mean Scores:")
for metric, score in mean_scores.items():
    print(f"{metric}: {score:.4f}")

# %% [code]
import matplotlib.pyplot as plt

# Fit the Extra Trees classifier to the training data
extra_trees_model.fit(X_train_scaled, oversampled_Y)

# Get feature importances
importances = extra_trees_model.feature_importances_

# Get feature names
feature_names = X_train_scaled.columns

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Extra Trees)")
plt.bar(range(X_train_scaled.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_scaled.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X_train_scaled.shape[1]])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# %% [code]
import lightgbm as lgb
# Create a LightGBM classifier
params = {'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 1.0,
 'importance_type': 'split',
 'learning_rate': 0.1,
 'max_depth': -1,
 'min_child_samples': 20,
 'min_child_weight': 0.001,
 'min_split_gain': 0.0,
 'n_estimators': 100,
 'n_jobs': -1,
 'num_leaves': 31,
 'objective': None,
 'random_state': 8057,
 'reg_alpha': 0.0,
 'reg_lambda': 0.0,
 'subsample': 1.0,
 'subsample_for_bin': 200000,
 'subsample_freq': 0}
lgb_model = lgb.LGBMClassifier(**params)
lgb_score = cross_validate(lgb_model, X=X_train_scaled, y=oversampled_Y, cv=10, scoring=scoring,return_train_score=True)
print(lgb_score)

# Compute the mean scores
mean_scores = {metric: np.mean(lgb_score[f'test_{metric}']) for metric in scoring}

print("Mean Scores:")
for metric, score in mean_scores.items():
    print(f"{metric}: {score:.4f}")

# %% [code]
# Fit the LightGBM classifier to the training data
lgb_model.fit(X_train_scaled, oversampled_Y)

# Get feature importances
importances = lgb_model.feature_importances_

# Get feature names
feature_names = X_train_scaled.columns

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances (LightGBM)")
plt.bar(range(X_train_scaled.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_scaled.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X_train_scaled.shape[1]])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# %% [code]
from sklearn.linear_model import LogisticRegression
# Create a Logistic Regression model
params = {'C': 1.0,
 'class_weight': None,
 'dual': False,
 'fit_intercept': True,
 'intercept_scaling': 1,
 'l1_ratio': None,
 'max_iter': 1000,
 'multi_class': 'auto',
 'n_jobs': None,
 'penalty': 'l2',
 'random_state': 8057,
 'solver': 'lbfgs',
 'tol': 0.0001,
 'verbose': 0,
 'warm_start': False}
lg_model = LogisticRegression(**params)
lg_score = cross_validate(lg_model, X=X_train_scaled, y=oversampled_Y, cv=10, scoring=scoring,return_train_score=False)
print(lg_score)
# Compute the mean scores
mean_scores = {metric: np.mean(lg_score[f'test_{metric}']) for metric in scoring}

print("Mean Scores:")
for metric, score in mean_scores.items():
    print(f"{metric}: {score:.4f}")

# %% [code]
# Fit the Logistic Regression model to the training data
lg_model.fit(X_train_scaled, oversampled_Y)

# Get feature importances
importances = np.abs(lg_model.coef_[0])

# Get feature names
feature_names = X_train_scaled.columns

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Logistic Regression)")
plt.bar(range(X_train_scaled.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_scaled.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X_train_scaled.shape[1]])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


# %% [code]
# For Logistic Regression
lg_model.fit(X_train_scaled, oversampled_Y)
y_pred_logistic = lg_model.predict(X_test_scaled)

# For Random Forest
rf_clf.fit(X_train_scaled, oversampled_Y)
y_pred_rf = rf_clf.predict(X_test_scaled)

# For Extra Trees
extra_trees_model.fit(X_train_scaled, oversampled_Y)
y_pred_extra_trees = extra_trees_model.predict(X_test_scaled)

# For LightGBM
lgb_model.fit(X_train_scaled, oversampled_Y)
y_pred_lgb = lgb_model.predict(X_test_scaled)

# %% [code]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, title):
    labels = ["No Diabetes", "Diabetes"]  # Define custom labels
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# %% [code]
#RFC
conf_matrix = confusion_matrix(y_test, y_pred_rf)

print("Confusion Matrix of Random Forest:")
print(conf_matrix)
plot_confusion_matrix(y_test, y_pred_rf, 'Confusion Matrix - Random Forest')


# %% [code]
#Extra Trees
conf_matrix = confusion_matrix(y_test, y_pred_extra_trees)

print("Confusion Matrix of Extra trees:")
print(conf_matrix)
plot_confusion_matrix(y_test, y_pred_extra_trees, 'Confusion Matrix - Extra Trees')

# %% [code]
#Light gradient boosting
conf_matrix = confusion_matrix(y_test, y_pred_lgb)

print("Confusion Matrix of Light gradient boosting:")
print(conf_matrix)
plot_confusion_matrix(y_test, y_pred_lgb, 'Confusion Matrix - LightGBM')

# %% [code]
# Logistic regression
conf_matrix = confusion_matrix(y_test, y_pred_logistic)

print("Confusion Matrix of Logistic Regression:")
print(conf_matrix)
plot_confusion_matrix(y_test, y_pred_logistic, 'Confusion Matrix - Logistic Regression')


# %% [code]
from sklearn.ensemble import VotingClassifier

# Create the ensemble classifier
ensemble_clf_rf_et = VotingClassifier(estimators=[('rf', rf_clf), ('et', extra_trees_model)], voting='soft')

# Fit the ensemble classifier
ensemble_clf_rf_et.fit(X_train_scaled, oversampled_Y)

# %% [code]
# Define the scoring metrics
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Perform cross-validation with the ensemble classifier
ensemble_clf_rf_et_score = cross_validate(ensemble_clf_rf_et, X=X_train_scaled, y=oversampled_Y, cv=10, scoring=scoring, return_train_score=False)
print(ensemble_clf_rf_et_score)

# Compute the mean scores
mean_scores = {metric: np.mean(ensemble_clf_rf_et_score[f'test_{metric}']) for metric in scoring}

print("Mean Scores:")
for metric, score in mean_scores.items():
    print(f"{metric}: {score:.4f}")

# %% [code]
# Ensembled
y_pred_ensemble_rf_et = ensemble_clf_rf_et.predict(X_test_scaled)

conf_matrix = confusion_matrix(y_test, y_pred_ensemble_rf_et)

print("Confusion Matrix of RF and et:")
print(conf_matrix)
plot_confusion_matrix(y_test, y_pred_ensemble_rf_et, 'Confusion Matrix - Ensembled Model')

