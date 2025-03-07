import os
import requests
import pandas as pd
import numpy as np
import datetime
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from dateparser import parse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import streamlit as st

HISTORICAL_DATA_FILE = "historical_data.csv"

# Step 1: Scraping with Pagination and Data Caching
def scrape_tutti_bikes(base_url, max_pages=5):
    """Scrapes bike listings from Tutti and returns structured data."""
    bike_data = []
    for page in range(1, max_pages + 1):
        response = requests.get(f"{base_url}?page={page}")
        if response.status_code != 200:
            break

        soup = BeautifulSoup(response.text, 'html.parser')
        listings = soup.find_all("div", class_="listing")
        if not listings:
            break

        for listing in listings:
            title = listing.find("h2").text.strip() if listing.find("h2") else "No Title"
            price = listing.find("span", class_="price").text.strip() if listing.find("span", class_="price") else None
            location = listing.find("span", class_="location").text.strip() if listing.find("span", class_="location") else "Unknown"
            date_posted = listing.find("time").text.strip() if listing.find("time") else None
            date_posted = parse(date_posted) if date_posted else None

            if price:
                price = float(price.replace("CHF", "").replace(",", "").strip())

            bike_data.append({
                "title": title,
                "price": price,
                "location": location,
                "date_posted": date_posted
            })

    return bike_data

# Step 2: Data Preparation & Historical Data Handling
def prepare_data(new_data):
    """Cleans and processes data, comparing with historical records."""
    df = pd.DataFrame(new_data)
    df["days_posted"] = (datetime.datetime.today() - df["date_posted"]).dt.days.fillna(0)
    df.drop(columns=["date_posted"], inplace=True)
    df.dropna(inplace=True)

    if os.path.exists(HISTORICAL_DATA_FILE):
        historical_df = pd.read_csv(HISTORICAL_DATA_FILE)
        df = pd.concat([historical_df, df]).drop_duplicates().reset_index(drop=True)

    df.to_csv(HISTORICAL_DATA_FILE, index=False)
    return df

# Step 3: Machine Learning Model
def train_model(df):
    df["sold"] = df["days_posted"].apply(lambda x: 1 if x < 30 else 0)
    X = df[["price", "days_posted"]]
    y = df["sold"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))
    return model

# Step 4: Visualization
def visualize_data(df):
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["price"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Days Posted vs. Sold")
    fig, ax = plt.subplots()
    sns.boxplot(x='sold', y='days_posted', data=df, ax=ax)
    st.pyplot(fig)

# Step 5: Streamlit UI
def main():
    st.title("Tutti Bike Sale Predictor")
    base_url = "https://www.tutti.ch/de/q/motorraeder/Ak8CrbW90b3JjeWNsZXOUwMDAwA?sorting=newest"
    
    if st.button("Scrape Latest Listings"):
        new_data = scrape_tutti_bikes(base_url)
        df = prepare_data(new_data)
        model = train_model(df)
        pickle.dump(model, open("model.pkl", "wb"))
        st.success("Data scraped and model updated!")
    
    model = pickle.load(open("model.pkl", "rb")) if os.path.exists("model.pkl") else None
    if model:
        st.sidebar.header("Make a Prediction")
        price = st.sidebar.number_input("Enter Price (CHF)", min_value=100, max_value=50000, value=500)
        days_posted = st.sidebar.slider("Days Since Posted", 0, 90, 10)
        if st.sidebar.button("Predict Sale Probability"):
            prediction = model.predict_proba([[price, days_posted]])[0][1]
            st.sidebar.write(f"Sale Probability: {prediction:.2%}")
            visualize_data(pd.read_csv(HISTORICAL_DATA_FILE))

if __name__ == "__main__":
    main()
