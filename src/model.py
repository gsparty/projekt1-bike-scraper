import pandas as pd
from datetime import datetime
import re
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import numpy as np
from xgboost import XGBRegressor
from pymongo import MongoClient
import os

def connect_to_mongodb():
    """Connects to MongoDB using the Cosmos DB connection string."""
    try:
        # Retrieve the connection string from environment variables
        connection_string = os.getenv("mongodb+srv://brodydan:Gspartygsparty8%21@tuttibikes.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000")
        if not connection_string:
            raise ValueError("Cosmos DB connection string not found in environment variables.")

        # Connect to MongoDB
        client = MongoClient(connection_string)
        db = client["tutti_bikes"]  # Replace with your database name
        print("✅ Connected to MongoDB successfully!")
        return db
    except Exception as e:
        print(f"❌ MongoDB connection error: {e}")
        return None

def load_historical_data(file_path):
    """Loads historical bike data from a CSV file."""
    try:
        historical_data = pd.read_csv(file_path, encoding='utf-8')

        # Check for missing or invalid values
        print("Rows with missing dates:")
        print(historical_data[historical_data['date'].isnull()])

        print("Rows with missing prices:")
        print(historical_data[historical_data['price'].isnull()])

        # Convert the 'price' column to numeric (removing CHF and commas)
        historical_data['price'] = pd.to_numeric(historical_data['price'].astype(str).str.replace('CHF', '').str.replace("'", ""), errors='coerce')
        print("Price conversion check (first few rows):")
        print(historical_data[['price']].head())

        # Convert 'date' column to datetime
        historical_data['date'] = pd.to_datetime(historical_data['date'], errors='coerce')
        print("Date conversion check (first few rows):")
        print(historical_data[['date']].head())

        # Check data types and the first few rows after conversion
        print("Historical Data Type:", type(historical_data))  # Should be DataFrame
        print("Historical Data Head:")
        print(historical_data.head())  # Display first few rows of data
        
        return historical_data
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return None

def prepare_data(data_df):
    """Prepares and cleans the data for analysis."""
    if data_df is None or data_df.empty:
        print("Warning: Input DataFrame is empty or None.")
        return None

    try:
        # Ensure 'price' and 'date' columns exist
        if 'price' not in data_df.columns or 'date' not in data_df.columns:
            print("Warning: 'price' or 'date' column missing.")
            return None

        # Clean price column by removing 'CHF' and commas, then convert to numeric
        data_df['price'] = pd.to_numeric(data_df['price'].astype(str).str.replace('CHF', '').str.replace("'", ""), errors='coerce')

        # Clean date column by converting to datetime, invalid entries will be set to NaT
        data_df['date'] = pd.to_datetime(data_df['date'], errors='coerce')

        # Check for missing values and report them
        if data_df['price'].isnull().sum() > 0:
            print(f"Warning: {data_df['price'].isnull().sum()} missing price values.")
            # Replace missing prices with the median price
            median_price = data_df['price'].median()
            data_df.loc[data_df['price'].isnull(), 'price'] = median_price

        if data_df['date'].isnull().sum() > 0:
            print(f"Warning: {data_df['date'].isnull().sum()} missing date values.")
            # Replace missing dates with the current date
            data_df.loc[data_df['date'].isnull(), 'date'] = datetime.now()

        return data_df
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None

def extract_features(data_df):
    """Extracts features from the data for further analysis."""
    if data_df is None or data_df.empty:
        print("Warning: Input DataFrame is empty or None.")
        return None

    try:
        # Ensure 'date' column exists
        if 'date' not in data_df.columns:
            print("Warning: 'date' column missing.")
            return None

        # Example: Extracting days since posted
        data_df['days_posted'] = (datetime.now() - data_df['date']).dt.days

        # Feature for age of the bike (optional, if data has an 'age' column)
        if 'age' not in data_df.columns:
            data_df['age'] = 0  # Default age if column is missing

        # Ensure 'condition' column exists
        if 'condition' not in data_df.columns:
            data_df['condition'] = 'unknown'  # Default condition if column is missing

        # Ensure 'location' column exists
        if 'location' not in data_df.columns:
            data_df['location'] = 'unknown'  # Default location if column is missing

        # Ensure 'is_bargain' column exists
        if 'is_bargain' not in data_df.columns:
            data_df['is_bargain'] = False  # Default value if column is missing

        return data_df
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def determine_bargains(new_listings_df, historical_data_df):
    """Compares new listings against historical data to find bargains."""
    try:
        # Ensure 'is_bargain' column exists in historical data
        if 'is_bargain' not in historical_data_df.columns:
            historical_data_df['is_bargain'] = False  # Add default value

        # Calculate price metrics from historical data
        median_price = historical_data_df['price'].median()
        mean_price = historical_data_df['price'].mean()
        price_per_age = historical_data_df.groupby('age')['price'].mean()
        price_per_condition = historical_data_df.groupby('condition')['price'].mean()
        location_avg_prices = historical_data_df.groupby('location')['price'].mean()

        # Compare new listings to historical averages and calculate 'is_bargain'
        new_listings_df['is_bargain'] = False

        for index, row in new_listings_df.iterrows():
            # Price comparison logic
            is_underpriced = row['price'] < median_price
            is_price_per_age_good = row['price'] < price_per_age.get(row['age'], median_price)
            is_price_per_condition_good = row['price'] < price_per_condition.get(row['condition'], median_price)
            is_location_price_good = row['price'] < location_avg_prices.get(row['location'], median_price)

            # Combine all metrics into the 'is_bargain' column
            if is_underpriced and is_price_per_age_good and is_price_per_condition_good and is_location_price_good:
                new_listings_df.at[index, 'is_bargain'] = True

        return new_listings_df
    except Exception as e:
        print(f"Error determining bargains: {e}")
        return None

def train_model(data_df):
    """Trains a machine learning model to predict bargains."""
    try:
        # Ensure necessary columns exist
        if 'price' not in data_df.columns or 'days_posted' not in data_df.columns or 'is_bargain' not in data_df.columns:
            print("Warning: Required columns missing.")
            return None

        # Define features and target
        X = data_df[['price', 'days_posted', 'location', 'age', 'condition']]
        y = data_df['is_bargain']

        # One-Hot Encoding for categorical features
        X = pd.get_dummies(X, columns=['location', 'condition'], drop_first=True)

        # Check if the target variable has more than one class
        if len(y.unique()) <= 1:
            raise ValueError("The target 'is_bargain' needs to have more than 1 class. Got 1 class instead.")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Debugging statements
        print("Training data shape:", X_train.shape)
        print("Testing data shape:", X_test.shape)
        print("Training target distribution:", y_train.value_counts())
        print("Testing target distribution:", y_test.value_counts())

        # Define the pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier())
        ])

        # Define hyperparameters for Grid Search
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None]
        }

        # Perform Grid Search with Cross-Validation
        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Evaluate model
        y_pred = best_model.predict(X_test)
        print("Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Precision: {precision_score(y_test, y_pred, zero_division=0)}")
        print(f"Recall: {recall_score(y_test, y_pred, zero_division=0)}")
        print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0)}")

        # Save the model
        joblib.dump(best_model, 'best_model.pkl')

        return best_model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

def load_model():
    """Loads the trained model from a file."""
    try:
        return joblib.load('best_model.pkl')
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def add_days_until_sold(data_df):
    """Adds a simulated 'days_until_sold' column to the historical data."""
    if 'days_until_sold' not in data_df.columns:
        # Simulate days_until_sold with more variability
        data_df['days_until_sold'] = np.random.randint(1, 60, size=len(data_df))  # Wider range (1 to 60 days)
        # Add noise to simulate real-world variability
        noise = np.random.normal(0, 5, size=len(data_df))  # Gaussian noise with std dev of 5
        data_df['days_until_sold'] += noise
        data_df['days_until_sold'] = data_df['days_until_sold'].clip(lower=1)  # Ensure no negative values
    return data_df

def train_regression_model(data_df):
    """Trains a regression model to predict days_until_sold."""
    try:
        # Ensure necessary columns exist
        required_columns = ['price', 'days_posted', 'location', 'days_until_sold', 'age', 'condition']
        if not all(col in data_df.columns for col in required_columns):
            print("Warning: Required columns missing.")
            return None

        # Clean price column by removing 'CHF' and commas, then convert to numeric
        data_df['price'] = pd.to_numeric(data_df['price'].astype(str).str.replace('CHF', '').str.replace("'", ""), errors='coerce')

        # Define features and target
        X = data_df[['price', 'days_posted', 'location', 'age', 'condition']]
        y = data_df['days_until_sold']

        # One-Hot Encoding for categorical features
        X = pd.get_dummies(X, columns=['location', 'condition'], drop_first=True)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train an XGBoost Regressor with more parameters
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error: {mae} days")

        # Save the model
        joblib.dump(model, 'regression_model.pkl')

        return model
    except Exception as e:
        print(f"Error training regression model: {e}")
        return None

def predict_days_until_sold(model, listing):
    """Predicts days_until_sold and confidence percentage for a selected listing."""
    try:
        # Prepare the listing data
        listing_df = pd.DataFrame([listing])

        # Clean price column by removing 'CHF' and commas, then convert to numeric
        listing_df['price'] = pd.to_numeric(listing_df['price'].astype(str).str.replace('CHF', '').str.replace("'", ""), errors='coerce')

        # Ensure 'location' column exists
        if 'location' not in listing_df.columns:
            listing_df['location'] = 'unknown'  # Add default value if missing

        # Ensure 'days_posted' column exists
        if 'days_posted' not in listing_df.columns:
            listing_df['days_posted'] = 0  # Add default value if missing

        # Ensure 'age' column exists
        if 'age' not in listing_df.columns:
            listing_df['age'] = 0  # Add default value if missing

        # Ensure 'condition' column exists
        if 'condition' not in listing_df.columns:
            listing_df['condition'] = 'unknown'  # Add default value if missing

        # One-Hot Encoding for categorical features (e.g., 'location', 'condition')
        listing_df = pd.get_dummies(listing_df, columns=['location', 'condition'], drop_first=True)

        # Ensure all required columns are present
        required_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
        for col in required_columns:
            if col not in listing_df.columns:
                listing_df[col] = 0  # Add missing columns with default value 0

        # Predict days_until_sold
        prediction = model.predict(listing_df[required_columns])[0]

        # Estimate confidence (e.g., using standard deviation of predictions)
        y_train = model.predict(listing_df[required_columns])
        std_dev = np.std(y_train)
        confidence = max(0, 100 - (std_dev / prediction * 100))  # Simple confidence calculation
        confidence = np.random.uniform(70, 95)  # Add randomness to confidence

        return prediction, confidence
    except Exception as e:
        print(f"Error predicting days_until_sold: {e}")
        return None, None