from datetime import datetime
import requests
from bs4 import BeautifulSoup
import dateparser
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def scrape_tutti_bikes(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Fehler beim Abruf der Seite: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    bike_data = []
    listings = soup.find_all("div", class_="listing")  # Update class name if needed

    for listing in listings:
        title = listing.find("h2").text.strip() if listing.find("h2") else "Kein Titel"
        price = listing.find("span", class_="price").text.strip() if listing.find("span", class_="price") else "Preis unbekannt"
        location = listing.find("span", class_="location").text.strip() if listing.find("span", class_="location") else "Standort unbekannt"

        # Extract the date posted (e.g., "2 days ago", "1 month ago")
        date_posted = listing.find("span", class_="date-posted").text.strip() if listing.find("span", class_="date-posted") else "Unknown"

        # Convert relative date_posted into a datetime object using dateparser
        if date_posted != "Unknown":
            date_posted = dateparser.parse(date_posted)  # Use dateparser to handle relative dates
        else:
            date_posted = None  # If not available, set it to None

        bike_data.append({
            "title": title,
            "price": price,
            "location": location,
            "date_posted": date_posted
        })

    return bike_data

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
