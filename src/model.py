from datetime import datetime
import requests
from bs4 import BeautifulSoup
import dateparser
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def scrape_tutti_bikes(url):
    """Scrapes bike listings from Tutti and returns structured data."""
    bike_data = []
    page = 1

    while True:
        response = requests.get(f"{url}?page={page}")
        if response.status_code != 200:
            print(f"Error fetching page {page}: {response.status_code}")
            break

        soup = BeautifulSoup(response.text, 'html.parser')
        listings = soup.find_all("div", class_="listing")  # Adjust class name if needed

        if not listings:
            print("No more listings found.")
            break

        for listing in listings:
            title = listing.find("h2").text.strip() if listing.find("h2") else "No Title"
            price = listing.find("span", class_="price").text.strip() if listing.find("span", class_="price") else None
            location = listing.find("span", class_="location").text.strip() if listing.find("span", class_="location") else "Unknown"
            date_posted = listing.find("time").text.strip() if listing.find("time") else None

            # Debugging log
            if price is None:
                print("Warning: Missing price detected.")

            # Convert the date to a standard format
            try:
                date_posted = dateparser.parse(date_posted)
            except ValueError:
                date_posted = None

            bike_data.append({
                "title": title,
                "price": price,
                "location": location,
                "date_posted": date_posted
            })

        page += 1

    return bike_data

def prepare_data(bike_data):
    """Cleans and prepares the scraped bike data for modeling."""
    today = datetime.today()

    for bike in bike_data:
        # Calculate days posted
        if bike['date_posted']:
            bike['days_posted'] = (today - bike['date_posted']).days
        else:
            bike['days_posted'] = 0  # Default to 0 if no date is available

    # Convert to dataframe
    df = pd.DataFrame(bike_data)

    # Handle missing or invalid prices
    df['price'] = (df['price']
                   .str.replace('CHF', '', regex=False)
                   .str.replace(',', '', regex=False)
                   .str.extract(r'(\d+)', expand=False)  # Fixed invalid escape sequence
                   .astype(float, errors='ignore'))
    df['price'].fillna(df['price'].median(), inplace=True)  # Fill missing prices with median

    return df

def extract_features(df):
    """Adds additional useful features to the data."""
    # Extract bike type if keywords like "mountain", "road" exist
    df['bike_type'] = df['title'].apply(lambda x: 
        'Mountain' if 'mountain' in x.lower() else 
        'Road' if 'road' in x.lower() else 
        'Other')

    # Add dummy variables for bike types
    df = pd.get_dummies(df, columns=['bike_type'], drop_first=True)

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
    """Predicts the probability of a sale."""
    prediction_prob = model.predict_proba([[price, days_posted]])[0]
    return prediction_prob[1]  # Probability of being sold

def visualize_trends(df):
    """Visualizes price vs. likelihood of selling."""
    # Correlation heatmap
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Price distribution
    sns.histplot(df['price'], bins=20, kde=True)
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.show()

    # Days posted vs. sold
    sns.boxplot(x='sold', y='days_posted', data=df)
    plt.title("Days Posted vs. Sold")
    plt.xlabel("Sold Status")
    plt.ylabel("Days Posted")
    plt.show()
