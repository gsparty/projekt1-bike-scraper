from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define models and hyperparameters
models = {
    'RandomForest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'NeuralNetwork': MLPRegressor()
}

params = {
    'RandomForest': {'n_estimators': [100, 200, 300, 400], 'max_depth': [None, 10, 20, 30, 40]},
    'XGBoost': {'n_estimators': [100, 200, 300, 400], 'learning_rate': [0.01, 0.05, 0.1, 0.2]},
    'NeuralNetwork': {'hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)], 'alpha': [0.0001, 0.001, 0.01, 0.1]}
}

# Assuming X and y are your features and target variable
# Replace the following lines with your actual data loading code
X = np.random.rand(100, 10)  # Example feature data
y = np.random.rand(100)  # Example target data (continuous values)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Perform Randomized Search
best_models = {}
for model_name in models:
    random_search = RandomizedSearchCV(models[model_name], params[model_name], cv=5, scoring='neg_mean_squared_error', n_iter=50, random_state=42)
    random_search.fit(X_train, y_train)
    best_models[model_name] = random_search.best_estimator_

# Evaluate models
for model_name in best_models:
    y_pred = best_models[model_name].predict(X_test)
    print(f"{model_name} Performance:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R2 Score: {r2_score(y_test, y_pred)}")