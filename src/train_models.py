import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
df = pd.read_csv("preprocessed_data.csv")

# Define features & target
features = ["title_length", "desc_length", "price"]  # Simple features for now
X = df[features]
y = df["high_price"]

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize price column (important for models like Logistic Regression)
scaler = StandardScaler()
X_train["price"] = scaler.fit_transform(X_train[["price"]])
X_test["price"] = scaler.transform(X_test[["price"]])

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

# Train & evaluate models
results = {}
for name, model in models.items():
    print(f"🚀 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Store results
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print(f"✅ {name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# Show best model
best_model = max(results, key=results.get)
print(f"\n🏆 Best Model: {best_model} with {results[best_model]:.4f} accuracy!")
