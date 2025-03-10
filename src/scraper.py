import requests
from bs4 import BeautifulSoup
import streamlit as st
import time
from pymongo import MongoClient

# Constants
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "de-DE,de;q=0.9",
    "Referer": "https://www.google.com",
    "DNT": "1",
    "Connection": "keep-alive"
}
BASE_URL = "https://www.tutti.ch/de/q/motorraeder/Ak8CrbW90b3JjeWNsZXOUwMDAwA?sorting=newest"
MAX_PAGES = 5
MAX_RETRIES = 3
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "tutti_bikes"
COLLECTION_NAME = "listings"

# MongoDB connection
def create_unique_index():
    """Creates a unique index on the 'url' field in MongoDB."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    collection.create_index("url", unique=True)
    st.info("✔️ Unique index on 'url' field created (if not already present).")

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
        try:
            collection.insert_one(item)
            new_data_count += 1
        except Exception as e:
            if "duplicate key error" in str(e):
                st.info(f"Duplicate entry skipped: {item.get('url', 'Unknown')}")
            else:
                st.error(f"Error saving entry: {e}")

    st.success(f"✅ {new_data_count} new listings saved to MongoDB.")
    if len(data) > new_data_count:
        st.info(f"ℹ️ {len(data) - new_data_count} duplicates were skipped.")

# Scraping Function
def scrape_tutti_bikes():
    """Scrapes bike listings from Tutti.ch."""
    bike_data = []
    session = requests.Session()
    session.headers.update(HEADERS)
    unique_urls = set()  # Track unique URLs

    for attempt in range(MAX_RETRIES):
        for page in range(1, MAX_PAGES + 1):
            url = f"{BASE_URL}&page={page}"

            try:
                st.write(f"📡 Fetching: {url}")
                response = session.get(url, timeout=10)

                if response.status_code != 200:
                    st.warning(f"⚠️ Error {response.status_code}: Unable to fetch page {page}")
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                listings = soup.find_all("div", class_="MuiBox-root mui-style-1haxbqe")

                if not listings:
                    st.warning(f"⚠️ No listings found on page {page}. Possible class name changes?")
                    continue

                for listing in listings:
                    # Extract URL
                    link_tag = listing.find("a", href=True)
                    url = "https://www.tutti.ch" + link_tag["href"] if link_tag else None

                    if url and url not in unique_urls:
                        unique_urls.add(url)  # Mark URL as seen

                        # Extract other data fields
                        name = listing.text.strip() if listing else "No title found"
                        desc_parent = listing.find_next("div", class_="MuiBox-root mui-style-wkoz8z")
                        desc = desc_parent.text.strip() if desc_parent else "No description found"
                        price_span = listing.find_next("span", class_="MuiTypography-root MuiTypography-body1 mui-style-1e5o6ii")
                        price = price_span.text.strip() if price_span else "No price found"
                        date_place_span = listing.find_next("span", class_="MuiTypography-root MuiTypography-body1 mui-style-13hgjc4")
                        date_place = date_place_span.text.strip() if date_place_span else "No date & place found"

                        bike_data.append({
                            "url": url,  # Unique identifier
                            "title": name,
                            "description": desc,
                            "price": price,
                            "date_place": date_place
                        })

            except requests.RequestException as e:
                st.error(f"❌ Request failed for page {page}: {e}")
                continue

        if bike_data:
            return bike_data

        st.warning(f"⚠️ No listings found. Retrying in 10 seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
        time.sleep(10)

    st.error("❌ No listings found after retries. Exiting.")
    return None

# Streamlit UI
def main():
    st.title("Bike Scraper 🚴‍♂️")

    if st.button("Create Unique Index"):
        create_unique_index()

    if st.button("Start Scraping"):
        with st.spinner("Scraping in progress..."):
            data = scrape_tutti_bikes()

        if not data:
            st.warning("❌ No listings found. Check the website or try later.")
        else:
            st.success(f"✅ Scraped {len(data)} listings successfully!")
            save_to_mongo(data)
            st.write(data[:5])  # Show first 5 results

if __name__ == "__main__":
    main()
