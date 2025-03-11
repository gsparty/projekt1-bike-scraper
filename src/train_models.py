import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import joblib
import xgboost as xgb
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Load dataset
file_path = "bike_listings.csv"
try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Print dataset info for debugging
data.info()
print("Dataset columns:", data.columns.tolist())

# Check for missing expected columns
expected_features = ["price", "days_posted", "is_new", "is_bargain", "is_urgent", "season", "bike_type_Mountain", "bike_type_Road"]
missing_features = [col for col in expected_features if col not in data.columns]

if missing_features:
    print(f"Error: The following expected columns are missing from the dataset: {missing_features}")
    print("Check your CSV file for column names and formatting.")
    exit()

# Select Features
X = data[expected_features]

# Check for missing target columns
targets = {"classification": "sold", "regression": "days_to_sell"}
for key, target in targets.items():
    if target not in data.columns:
        print(f"Error: Target column '{target}' is missing from the dataset.")
        exit()

y_classification = data[targets["classification"]]
y_regression = data[targets["regression"]]

# Train/Test Split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_classification, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# Scaling Features
scaler = StandardScaler()
X_train_c = scaler.fit_transform(X_train_c)
X_test_c = scaler.transform(X_test_c)
X_train_r = scaler.fit_transform(X_train_r)
X_test_r = scaler.transform(X_test_r)

# Model 1: RandomForest Classification (Probability of Sale)
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train_c, y_train_c)
y_pred_c_rf = clf_rf.predict(X_test_c)
y_pred_c_rf_proba = clf_rf.predict_proba(X_test_c)[:, 1] * 100

# Model 2: RandomForest Regression (Estimated Time-to-Sale)
reg_rf = RandomForestRegressor(n_estimators=100, random_state=42)
reg_rf.fit(X_train_r, y_train_r)
y_pred_r_rf = reg_rf.predict(X_test_r)

# Evaluation - Classification
print("RandomForest Classification Model Metrics:")
print("Accuracy:", accuracy_score(y_test_c, y_pred_c_rf))
print("Precision:", precision_score(y_test_c, y_pred_c_rf))
print("Recall:", recall_score(y_test_c, y_pred_c_rf))
print("F1 Score:", f1_score(y_test_c, y_pred_c_rf))

# Save Models
joblib.dump(clf_rf, "classification_model_rf.pkl")
joblib.dump(reg_rf, "regression_model_rf.pkl")
