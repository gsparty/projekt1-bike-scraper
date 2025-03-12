from datetime import datetime, timedelta
import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import numpy as np

def prepare_data(bike_data):
    """Cleans and prepares the scraped bike data for modeling."""
    today = datetime.today()

    for bike in bike_data:
        # Handle "heute" and "gestern" and extract zip code
        if bike['date']:
            if 'heute' in bike['date'].lower():
                bike['date'] = today.strftime('%d/%m/%Y')
            elif 'gestern' in bike['date'].lower():
                bike['date'] = (today - timedelta(days=1)).strftime('%d/%m/%Y')

            # Extract zip code from the date string (if present)
            zip_code_match = re.search(r'\b\d{4}\b', bike['date'])
            if zip_code_match:
                bike['place'] = f"{bike['place']} {zip_code_match.group()}"
                bike['date'] = re.sub(r'\b\d{4}\b', '', bike['date']).strip()

            # Parse the date and calculate days posted
            parsed_date = pd.to_datetime(bike['date'], dayfirst=True)
            bike['days_posted'] = (today - parsed_date).days if parsed_date else None
        else:
            bike['days_posted'] = None  # Default to None if no date is available

    # Convert to dataframe
    df = pd.DataFrame(bike_data)

    # Handle missing or invalid prices
    df['price'] = (df['price']
                   .astype(str)
                   .str.replace('CHF', '', regex=False)
                   .str.replace(',', '', regex=False)
                   .str.extract(r'(\d+)', expand=False)  # Extract numeric part
                   .astype(float))

    # Fill missing prices with median price
    df['price'].fillna(df['price'].median(), inplace=True)

    return df

def extract_features(df):
    """Adds additional useful features to the data."""
    
    # Extract bike type based on title keywords
    df['bike_type'] = df['title'].apply(lambda x: 
        'Mountain' if 'mountain' in x.lower() else 
        'Road' if 'road' in x.lower() else 
        'Other')

    # Convert bike types into dummy variables (one-hot encoding)
    df = pd.get_dummies(df, columns=['bike_type'], drop_first=True)

    # Extract numerical values from price
    df['price'] = df['price'].apply(lambda x: float(re.sub(r'[^\d.]', '', str(x))) if pd.notnull(x) else x)

    # Analyze description for key selling terms
    df['is_new'] = df['description'].apply(lambda x: 1 if isinstance(x, str) and 'new' in x.lower() else 0)
    df['is_bargain'] = df['description'].apply(lambda x: 1 if isinstance(x, str) and 'bargain' in x.lower() else 0)
    df['is_urgent'] = df['description'].apply(lambda x: 1 if isinstance(x, str) and 'urgent' in x.lower() else 0)

    # Time-based features (extract season from date)
    df['season'] = df['date'].apply(lambda x: 
        (pd.to_datetime(x, dayfirst=True).month % 12 // 3 + 1) if isinstance(x, str) and pd.to_datetime(x, dayfirst=True) is not None else None)

    # Extract day of the week from date
    df['day_of_week'] = df['date'].apply(lambda x: 
        pd.to_datetime(x, dayfirst=True).dayofweek if isinstance(x, str) and pd.to_datetime(x, dayfirst=True) is not None else None)

    # Extract hour of the day from date
    df['hour_of_day'] = df['date'].apply(lambda x: 
        pd.to_datetime(x, dayfirst=True).hour if isinstance(x, str) and pd.to_datetime(x, dayfirst=True) is not None else None)

    # Normalize numerical features
    scaler = StandardScaler()
    df[['price', 'days_posted']] = scaler.fit_transform(df[['price', 'days_posted']])

    return df

def determine_bargains(df, historical_data):
    """Determines if a listing is a bargain based on historical data."""
    # Train a regression model to predict expected price
    features = ['condition', 'age', 'location', 'other_features']
    X = historical_data[features]
    y = historical_data['price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    df['expected_price'] = model.predict(df[features])
    df['residuals'] = df['price'] - df['expected_price']

    # Determine bargains
    threshold = -500  # Example threshold for bargain classification
    df['is_bargain'] = df['residuals'].apply(lambda x: 1 if x < threshold else 0)

    return df

def select_and_evaluate_model(X, y):
    """Selects and evaluates the best model using Randomized Search and K-Fold Cross-Validation."""
    
    models = {
        'XGBoost': XGBRegressor()
    }

    params = {
        'XGBoost': {'n_estimators': [100, 200, 300, 400], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}
    }

    best_models = {}
    
    for model_name in models:
        random_search = RandomizedSearchCV(models[model_name], params[model_name], cv=5, scoring='neg_mean_squared_error', n_iter=50, random_state=42)
        random_search.fit(X, y)
        best_models[model_name] = random_search.best_estimator_

        print(f"Best parameters for {model_name}: {random_search.best_params_}")

        kf = KFold(n_splits=5)
        mse_scores = []
        r2_scores = []

        for train_index, test_index in kf.split(X):
            X_train_kf, X_test_kf = X[train_index], X[test_index]
            y_train_kf, y_test_kf = y[train_index], y[test_index]

            best_models[model_name].fit(X_train_kf, y_train_kf)
            y_pred_kf = best_models[model_name].predict(X_test_kf)

            mse_scores.append(mean_squared_error(y_test_kf, y_pred_kf))
            r2_scores.append(r2_score(y_test_kf, y_pred_kf))

        print(f"{model_name} Performance:")
        print(f"Mean Squared Error: {np.mean(mse_scores)}")
        print(f"R2 Score: {np.mean(r2_scores)}")

# Example usage:
# historical_data should be a DataFrame containing historical listings with features and prices.
# new_listings should be a DataFrame containing new listings to be evaluated.

# Prepare historical data and new listings data
historical_data_df = prepare_data(historical_data)
new_listings_df = prepare_data(new_listings)

# Extract features from historical data and new listings data
historical_data_df = extract_features(historical_data_df)
new_listings_df = extract_features(new_listings_df)

# Determine bargains in new listings based on historical data
new_listings_df_with_bargains = determine_bargains(new_listings_df, historical_data_df)

# Select and evaluate the best model using historical data
X = historical_data_df.drop(columns=['price'])
y = historical_data_df['price']
select_and_evaluate_model(X, y)