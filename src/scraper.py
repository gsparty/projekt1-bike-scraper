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
        # Use the URL as a unique identifier
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

# Scraping Function
def scrape_tutti_bikes():
    """Scrapes bike listings from Tutti.ch."""
    bike_data = []
    session = requests.Session()
    session.headers.update(HEADERS)

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
                    # Extract link (Unique identifier for listings)
                    link = listing.find("a")
                    listing_url = "https://www.tutti.ch" + link["href"] if link else "No URL found"

                    # Extract title
                    title_element = listing.find("div", class_="MuiBox-root mui-style-1haxbqe")
                    title = title_element.text.strip() if title_element else "No title found"

                    # Extract description
                    desc_parent = listing.find("div", class_="MuiBox-root mui-style-wkoz8z")
                    desc_element = desc_parent.find("span", class_="MuiTypography-root MuiTypography-body1 mui-style-1e5o6ii") if desc_parent else None
                    description = desc_element.text.strip() if desc_element else "No description found"

                    # Extract price
                    price_container = listing.find("div", class_="MuiBox-root mui-style-1haxbqe")  # Adjust this class if needed
                    price_element = price_container.find("span", class_="MuiTypography-root MuiTypography-body1 mui-style-1e5o6ii") if price_container else None
                    price = price_element.text.strip() if price_element else "No price found"

                    # Extract date & place
                    date_place_element = listing.find("span", class_="MuiTypography-root MuiTypography-body1 mui-style-13hgjc4")
                    date_place = date_place_element.text.strip() if date_place_element else "No date & place found"

                    # Separate date and place
                    date_place_parts = date_place.split(", ")
                    place = date_place_parts[0] if len(date_place_parts) > 0 else "No place found"
                    date = ", ".join(date_place_parts[1:]) if len(date_place_parts) > 1 else "No date found"

                    bike_data.append({
                        "url": listing_url,
                        "title": title,
                        "description": description,
                        "price": price,
                        "place": place,
                        "date": date
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

    if st.button("Check IP"):
        check_ip()

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