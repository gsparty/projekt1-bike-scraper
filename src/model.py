import pandas as pd
from datetime import datetime
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib

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
        historical_data['price'] = pd.to_numeric(historical_data['price'].astype(str).str.replace('CHF', '').str.replace(',', ''), errors='coerce')
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
        data_df['price'] = pd.to_numeric(data_df['price'].astype(str).str.replace('CHF', '').str.replace(',', ''), errors='coerce')

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
        # Checking if both dataframes are not empty
        if new_listings_df.empty or historical_data_df.empty:
            raise ValueError("Both new data and historical data must be non-empty")

        # Calculating price metrics from historical data
        median_price = historical_data_df['price'].median()
        mean_price = historical_data_df['price'].mean()
        price_per_age = historical_data_df.groupby('age')['price'].mean()
        price_per_condition = historical_data_df.groupby('condition')['price'].mean()
        location_avg_prices = historical_data_df.groupby('location')['price'].mean()

        # Compare new listings to historical averages and calculate 'is_bargain'
        new_listings_df['is_bargain'] = False

        for index, row in new_listings_df.iterrows():
            # Price comparison logic: check if the price is below the median, and factor in age, condition, and location
            is_underpriced = row['price'] < median_price
            is_price_per_age_good = row['price'] < price_per_age.get(row['age'], median_price)
            is_price_per_condition_good = row['price'] < price_per_condition.get(row['condition'], median_price)
            is_location_price_good = row['price'] < location_avg_prices.get(row['location'], median_price)

            # Combine all metrics into the 'is_bargain' column
            if is_underpriced and is_price_per_age_good and is_price_per_condition_good and is_location_price_good:
                new_listings_df.at[index, 'is_bargain'] = True

        # Ensure there are both positive and negative samples
        if new_listings_df['is_bargain'].sum() == 0 or new_listings_df['is_bargain'].sum() == len(new_listings_df):
            raise ValueError("The target 'is_bargain' needs to have more than 1 class. Got 1 class instead.")

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
        X = data_df[['price', 'days_posted', 'location']]
        y = data_df['is_bargain']

        # One-Hot Encoding for categorical features
        X = pd.get_dummies(X, columns=['location'], drop_first=True)

        # Check if the target variable has more than one class
        if len(y.unique()) <= 1:
            raise ValueError("The target 'is_bargain' needs to have more than 1 class. Got 1 class instead.")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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