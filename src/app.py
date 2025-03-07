from model import create_model, predict_selling_probability, scrape_tutti_bikes  # Import scrape_tutti_bikes
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import datetime

def prepare_data(bike_data):
    """Prepares the scraped bike data for modeling."""
    today = datetime.datetime.today()

    for bike in bike_data:
        if bike['date_posted']:
            bike['days_posted'] = (today - bike['date_posted']).days
        else:
            bike['days_posted'] = 0  # Default to 0 if no date is available

    df = pd.DataFrame(bike_data)
    df['price'] = df['price'].str.replace('CHF', '').str.replace(',', '').astype(float, errors='ignore')
    df['price'] = df['price'].fillna(df['price'].median())

    return df

def create_model(df):
    """Trains a machine learning model to predict the chance of selling."""
    df['sold'] = df['days_posted'].apply(lambda x: 1 if x < 30 else 0)  # Bikes sold within 30 days

    X = df[['price', 'days_posted']]
    y = df['sold']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model

def predict_selling_probability(model, price, days_posted):
    """Predicts the chance of a sale based on price and days posted."""
    prediction = model.predict([[price, days_posted]])
    return prediction[0]  # 0 = not sold, 1 = sold

if __name__ == "__main__":
    # Example usage
    bike_data = scrape_tutti_bikes("https://www.tutti.ch/de/li/ganze-schweiz/velo")  # Now using the imported function
    df = prepare_data(bike_data)
    model = create_model(df)
    
    # Test prediction with a sample
    test_price = 500  # Example price
    test_days_posted = 20  # Example days posted
    sale_chance = predict_selling_probability(model, test_price, test_days_posted)
    print(f"Prediction: {'Sold' if sale_chance == 1 else 'Not Sold'}")
