import streamlit as st
from pymongo import MongoClient
from scraper import scrape_tutti_bikes
from model import prepare_data, extract_features
import requests

# Constants
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "tutti_bikes"
COLLECTION_NAME = "listings"
BASE_URL = "https://www.tutti.ch/de/q/motorraeder/Ak8CrbW90b3JjeWNsZXOUwMDAwA?sorting=newest"

# MongoDB connection
def save_to_mongo(data):
    """Saves scraped data to MongoDB while avoiding duplicates."""
    if not data:
        st.warning("⚠️ No data to save.")
        return

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    new_data_count = 0
    for item in data:
        if not collection.find_one({"url": item["url"]}):
            collection.insert_one(item)
            new_data_count += 1

    st.success(f"✅ {new_data_count} new listings saved to MongoDB.")
    if len(data) > new_data_count:
        st.info(f"ℹ️ {len(data) - new_data_count} duplicates were skipped.")

# Debugging function
def check_ip():
    """Displays the public IP address."""
    try:
        response = requests.get("https://api64.ipify.org?format=json")
        st.write(f"🌍 Your Public IP: {response.json()['ip']}")
    except Exception as e:
        st.error(f"❌ Failed to fetch public IP: {e}")

# Streamlit UI
def main():
    st.title("Bike Scraper 🚴‍♂️")

    if st.button("Check IP"):
        check_ip()

    if st.button("Start Scraping"):
        with st.spinner("Scraping in progress..."):
            data = scrape_tutti_bikes(BASE_URL, max_pages=5)
            st.write(f"Scraped {len(data)} listings.")

        if not data:
            st.warning("❌ No listings found. Check the website or try later.")
        else:
            st.success(f"✅ Scraped {len(data)} listings successfully!")
            save_to_mongo(data)
            st.write(data[:5])  # Show first 5 results

            # Prepare and visualize data
            df = prepare_data(data)
            df = extract_features(df)
            if '_id' in df.columns:
                df = df.drop(columns=['_id'])
            st.write(df.head())  # Show first few rows of the dataframe excluding '_id'

if __name__ == "__main__":
    main()