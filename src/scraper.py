import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import streamlit as st
from pymongo import MongoClient
import pandas as pd
from pymongo.errors import PyMongoError

# Constants
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
BASE_URL = "https://www.tutti.ch/de/q/motorraeder/Ak8CrbW90b3JjeWNsZXOUwMDAwA?sorting=newest"
MAX_PAGES = 5

# Get MongoDB URI from environment variables; defaults to localhost for local testing.
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "tutti_bikes"
COLLECTION_NAME = "listings"

def save_to_mongo(data):
    """Saves scraped data to MongoDB while avoiding duplicates."""
    if not data:
        st.warning("⚠️ No data to save.")
        return

    client = None
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        new_data_count = 0
        existing_urls = {item["url"] for item in collection.find({}, {"url": 1, "_id": 0})}
        
        new_items = [item for item in data if item["url"] not in existing_urls]
        
        if new_items:
            for item in new_items:
                # Convert date to datetime
                if item["date"] != "No date":
                    try:
                        item["date"] = datetime.strptime(item["date"], "%d.%m.%Y")
                    except ValueError:
                        item["date"] = None
                
                # Ensure price is integer or null
                if item["price"]:
                    try:
                        item["price"] = int(item["price"])
                    except (ValueError, TypeError):
                        item["price"] = None

            result = collection.insert_many(new_items)
            new_data_count = len(result.inserted_ids)
        
        st.success(f"✅ {new_data_count} new listings saved to MongoDB.")
        st.info(f"ℹ️ {len(data) - new_data_count} duplicates skipped")

    except PyMongoError as e:
        st.error(f"❌ MongoDB error: {e}")
    finally:
        if client:
            client.close()

def convert_relative_date(date_str):
    """Converts relative dates to actual dates with validation"""
    today = datetime.today()
    try:
        if "Heute" in date_str:
            return today.strftime("%d.%m.%Y")
        if "Gestern" in date_str:
            return (today - timedelta(days=1)).strftime("%d.%m.%Y")
        # Validate absolute dates
        date_obj = datetime.strptime(date_str.split()[0], "%d.%m.%Y")
        return date_obj.strftime("%d.%m.%Y")
    except:
        return "No date"

def extract_price(price_str):
    """Precision price extraction with currency context awareness"""
    if not price_str:
        return None
    try:
        # Remove apostrophes and other punctuation, then extract digits
        price_numbers = re.findall(r'\d+', price_str.replace("'", ""))
        return int("".join(price_numbers)) if price_numbers else None
    except (ValueError, AttributeError, TypeError):
        return None

def parse_location_date(element):
    """Improved location/date parsing"""
    if not element:
        return "No location", "No date"
    
    text = element.get_text(strip=True)
    parts = text.rsplit(', ', 1)
    
    if len(parts) == 2:
        location = parts[0].replace(', ', ' ')
        date_time = parts[1]
    else:
        location = text
        date_time = "No date"
    
    return location, convert_relative_date(date_time)

def scrape_tutti_bikes(url, max_pages=5):
    bike_data = []
    page = 1

    while page <= max_pages:
        try:
            response = requests.get(f"{url}&o={(page-1)*30}", headers=HEADERS)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            listings = soup.find_all("div", class_="mui-style-qlw8p1")

            if not listings:
                st.warning(f"No listings found on page {page}")
                break

            for listing in listings:
                try:
                    # URL handling
                    link_element = listing.find("a", href=True)
                    listing_url = f"https://www.tutti.ch{link_element['href']}" if link_element else None

                    # Title
                    title_element = listing.find("div", class_="mui-style-1haxbqe")
                    title = title_element.get_text(strip=True) if title_element else "No title"

                    # Description
                    desc_element = listing.find("span", class_="mui-style-kw4z3u")
                    description = desc_element.get_text(strip=True) if desc_element else "No description"

                    # Price extraction
                    price = None
                    
                    # First method: Official price element (old model approach)
                    price_container = listing.find("div", class_="MuiBox-root mui-style-1fhgjcy")
                    if price_container:
                        price_span = price_container.find("span")
                        price_str = price_span.get_text(strip=True) if price_span else None
                        price = extract_price(price_str) if price_str else None

                    # Second method: Search description for price patterns
                    if not price and description:
                        price_match = re.search(
                            r'(?:CHF|Preis|Prix|Price)[^\d]*(\d+(?:[\s\']?\d+)*)', 
                            description, 
                            re.IGNORECASE
                        )
                        if price_match:
                            price = extract_price(price_match.group(0))

                    # Location and Date
                    date_place_element = listing.find("span", class_="mui-style-1xafick")
                    place, date = parse_location_date(date_place_element)

                    # Image
                    image_element = listing.find("img")
                    image_url = image_element.get("src", None) if image_element else None

                    # Days posted calculation
                    days_posted = None
                    if date != "No date":
                        try:
                            date_obj = datetime.strptime(date, "%d.%m.%Y")
                            today = datetime.today()
                            delta = (today - date_obj).days
                            days_posted = max(delta, 0) if delta < 365*5 else None
                        except ValueError:
                            days_posted = None

                    bike_data.append({
                        "url": listing_url,
                        "title": title,
                        "description": description,
                        "price": price,
                        "place": place,
                        "date": date,
                        "days_posted": days_posted,
                        "image": image_url
                    })

                except Exception as e:
                    st.error(f"Error processing listing: {str(e)}")
                    continue

            st.info(f"✅ Successfully scraped page {page} with {len(listings)} listings")
            page += 1

        except requests.HTTPError as e:
            st.error(f"HTTP error fetching page {page}: {str(e)}")
            break
        except Exception as e:
            st.error(f"General error on page {page}: {str(e)}")
            break

    return bike_data

def main():
    st.title("Bike Scraper 🚴♂️")

    if st.button("Start Scraping"):
        with st.spinner("Scraping in progress..."):
            data = scrape_tutti_bikes(BASE_URL, MAX_PAGES)
            
            if not data:
                st.warning("❌ No listings found. Check the website or try later.")
                return
                
            save_to_mongo(data)
            
            # Display results from MongoDB (excluding _id)
            client = MongoClient(MONGO_URI)
            db = client[DB_NAME]
            collection = db[COLLECTION_NAME]
            
            # Convert cursor to list of dicts and exclude _id
            data_from_db = list(collection.find({}, {"_id": 0}))
            df = pd.DataFrame(data_from_db)
            
            # Format display
            if not df.empty:
                display_df = df[["title", "price", "place", "date", "days_posted"]].copy()
                display_df["price"] = display_df["price"].apply(
                    lambda x: f"CHF {x:,}" if pd.notnull(x) else "N/A"
                )
                display_df["date"] = pd.to_datetime(display_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                
                st.success(f"✅ Successfully processed {len(df)} listings")
                st.dataframe(display_df.head(10))
            else:
                st.warning("No data to display")

if __name__ == "__main__":
    main()
