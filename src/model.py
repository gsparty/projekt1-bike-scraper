import pandas as pd
from datetime import datetime
import re

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
        historical_data['price'] = pd.to_numeric(historical_data['price'].str.replace('CHF', '').str.replace(',', ''), errors='coerce')
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
    try:
        # Clean price column by removing 'CHF' and commas, then convert to numeric
        data_df['price'] = pd.to_numeric(data_df['price'].str.replace('CHF', '').str.replace(',', ''), errors='coerce')

        # Clean date column by converting to datetime, invalid entries will be set to NaT
        data_df['date'] = pd.to_datetime(data_df['date'], errors='coerce')

        # Check for missing values and report them
        if data_df['price'].isnull().sum() > 0:
            print(f"Warning: {data_df['price'].isnull().sum()} missing price values.")
            # Optional: Replace missing prices with a default value or drop rows
            data_df['price'].fillna(data_df['price'].median(), inplace=True)  # Fills with the median price

        if data_df['date'].isnull().sum() > 0:
            print(f"Warning: {data_df['date'].isnull().sum()} missing date values.")
            # Optional: Replace missing dates with a default date or drop rows
            data_df['date'].fillna(datetime.now(), inplace=True)  # Fills with the current date

        return data_df
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None


def extract_features(data_df):
    """Extracts features from the data for further analysis."""
    try:
        # Example: Extracting days since posted
        data_df['days_posted'] = (datetime.now() - data_df['date']).dt.days

        # Feature for age of the bike (optional, if data has an 'age' column)
        data_df['age'] = data_df['age'].fillna(0).astype(int)  # Assuming 'age' column exists

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

        return new_listings_df
    except Exception as e:
        print(f"Error determining bargains: {e}")
        return None
