import requests
from bs4 import BeautifulSoup
import streamlit as st
import dateparser
import time
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Constants
HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE_URL = "https://www.tutti.ch/de/q/motorraeder/Ak8CrbW90b3JjeWNsZXOUwMDAwA?sorting=newest"
MAX_PAGES = 15
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "tutti_bikes"
COLLECTION_NAME = "listings"

# Connect to MongoDB
def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

# Scraping Function
def scrape_tutti_bikes():
    bike_data = []
    session = requests.Session()
    session.headers.update(HEADERS)

    for page in range(1, MAX_PAGES + 1):
        url = f"{BASE_URL}&page={page}"
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            continue

        soup = BeautifulSoup(response.text, 'html.parser')
        listings = soup.find_all("div", class_="MuiBox-root mui-style-1haxbqe")
        
        for listing in listings:
            name = listing.text.strip() if listing else "No title"
            desc = listing.find_next("div", class_="MuiBox-root mui-style-wkoz8z").text.strip() if listing else "No description"
            price_span = listing.find_next("span", class_="MuiTypography-root MuiTypography-body1 mui-style-1e5o6ii")
            price = price_span.text.strip().replace("CHF", "").replace(".", "").strip() if price_span else "0"
            
            date_place_span = listing.find_next("span", class_="MuiTypography-root MuiTypography-body1 mui-style-13hgjc4")
            date_place = date_place_span.text.strip() if date_place_span else "No date"

            bike_data.append({
                "title": name,
                "description": desc,
                "price": int(price) if price.isdigit() else 0,
                "date_place": date_place
            })

    return bike_data

# Save to MongoDB
def save_to_mongo(data):
    if not data:
        return
    collection = get_mongo_collection()
    collection.insert_many(data)
    st.success(f"✅ {len(data)} listings saved to MongoDB.")

# Load Data from MongoDB
def load_data_from_mongo():
    collection = get_mongo_collection()
    return list(collection.find({}, {"_id": 0}))

# Train ML Model
def train_model():
    data = load_data_from_mongo()
    df = pd.DataFrame(data)
    if df.empty:
        st.warning("No data available for training.")
        return
    
    df["price"] = pd.to_numeric(df["price"], errors='coerce')
    df.dropna(inplace=True)
    
    X = df[["price"]]  # Placeholder for more features
    y = np.random.randint(0, 2, size=len(df))  # Simulated sold/not sold labels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    st.success("✅ Model trained and saved.")

# Streamlit UI
def main():
    st.title("Bike Scraper & ML Model 🚴‍♂️")
    if st.button("Scrape & Save Data"):
        data = scrape_tutti_bikes()
        save_to_mongo(data)
    
    if st.button("Train Model"):
        train_model()

if __name__ == "__main__":
    main()
