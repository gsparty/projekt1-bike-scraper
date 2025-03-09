import requests
from bs4 import BeautifulSoup
import streamlit as st
import dateparser
import time

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

# Scraping Function
def scrape_tutti_bikes():
    """Scrapes bike listings from Tutti.ch with proper headers and URL formatting."""
    bike_data = []
    session = requests.Session()

    for attempt in range(MAX_RETRIES):
        for page in range(1, MAX_PAGES + 1):
            url = f"{BASE_URL}&page={page}"  # Correct URL format
            session.headers.update(HEADERS)  # Reset headers each request

            st.write(f"📡 Fetching: {url}")  # Debugging: print requested URL

            try:
                response = session.get(url, timeout=10)
                st.write(f"🔍 Status Code: {response.status_code}")  # Debugging: Print status code
                st.write(f"🔄 Response URL: {response.url}")  # Debugging: See if it redirects

                if response.status_code == 404:
                    st.error(f"🚨 Page {page} not found (404). Possible bot detection.")
                    continue
                
                if response.status_code != 200:
                    st.warning(f"⚠️ Error fetching page {page}: {response.status_code}")
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                listings = soup.find_all("article", class_="sc-iBPRYJ")

                if not listings:
                    st.warning(f"⚠️ No listings found on page {page}.")
                    continue

                for listing in listings:
                    title = listing.find("h3").text.strip() if listing.find("h3") else "No Title"
                    price_elem = listing.find("span", class_="sc-eCssSg")
                    price = price_elem.text.strip() if price_elem else "N/A"
                    location_elem = listing.find("span", class_="sc-jrAGrp")
                    location = location_elem.text.strip() if location_elem else "Unknown"
                    date_elem = listing.find("span", class_="sc-fubCfw")
                    date_posted = dateparser.parse(date_elem.text.strip()) if date_elem else None

                    bike_data.append({
                        "title": title,
                        "price": price,
                        "location": location,
                        "date_posted": date_posted
                    })

            except requests.RequestException as e:
                st.error(f"❌ Request failed for page {page}: {e}")
                continue

        if bike_data:
            return bike_data

        st.write(f"⚠️ No listings found. Retrying in 10 seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
        time.sleep(10)

    st.error("❌ No listings found after retries. Exiting.")
    return None

# Streamlit UI
def main():
    st.title("Bike Scraper 🚴‍♂️")
    if st.button("Start Scraping"):
        with st.spinner("Scraping in progress..."):
            data = scrape_tutti_bikes()
        
        if not data:
            st.warning("❌ No listings found. Check the website or try later.")
        else:
            st.success(f"✅ Scraped {len(data)} listings successfully!")
            st.write(data[:5])  # Show first 5 results

if __name__ == "__main__":
    main()
