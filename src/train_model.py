import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your dataset (assuming it's stored in CSV)
data_path = "historical_data.csv"
df = pd.read_csv(data_path)

# Define features and target
features = ["price", "condition", "age", "days_posted"]
target = "bargain"  # Assuming you want to predict whether a bike is a bargain

# If 'bargain' is not in your dataset, create it based on a rule (e.g., sells in <30 days)
if target not in df.columns:
    df[target] = (df["days_posted"] < 30).astype(int)  # 1 = bargain, 0 = not

# Convert categorical data (e.g., condition) to numbers
df["condition"] = df["condition"].astype("category").cat.codes

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model as model.pkl
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl!")
