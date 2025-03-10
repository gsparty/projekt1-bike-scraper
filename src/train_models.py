import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
data = pd.read_csv("bike_listings.csv")  # Replace with actual dataset

# Data Cleaning (Remove Duplicates)
data = data.drop_duplicates()

# Feature Engineering
features = ["price", "location", "condition"]  # Adjust based on dataset
X = data[features]

# Target Variables
y_classification = data["sold"]  # Binary: 1 = Sold, 0 = Not Sold
y_regression = data["days_to_sell"]  # Number of days until sold

# Train/Test Split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_classification, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# Scaling Features
scaler = StandardScaler()
X_train_c = scaler.fit_transform(X_train_c)
X_test_c = scaler.transform(X_test_c)
X_train_r = scaler.fit_transform(X_train_r)
X_test_r = scaler.transform(X_test_r)

# Model 1: Classification (Probability of Sale)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)

# Model 2: Regression (Estimated Time-to-Sale)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)

# Evaluation - Classification
print("Classification Model Metrics:")
print("Accuracy:", accuracy_score(y_test_c, y_pred_c))
print("Precision:", precision_score(y_test_c, y_pred_c))
print("Recall:", recall_score(y_test_c, y_pred_c))
print("F1 Score:", f1_score(y_test_c, y_pred_c))

# Evaluation - Regression
print("\nRegression Model Metrics:")
print("MAE:", mean_absolute_error(y_test_r, y_pred_r))
print("RMSE:", mean_squared_error(y_test_r, y_pred_r, squared=False))
print("R2 Score:", r2_score(y_test_r, y_pred_r))

# Save Models
joblib.dump(clf, "classification_model.pkl")
joblib.dump(reg, "regression_model.pkl")
