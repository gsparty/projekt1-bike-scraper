from datetime import datetime, timedelta
import dateparser
import pandas as pd
import re

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
            parsed_date = dateparser.parse(bike['date'])
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
        (dateparser.parse(x).month % 12 // 3 + 1) if isinstance(x, str) and dateparser.parse(x) is not None else None)

    return df
