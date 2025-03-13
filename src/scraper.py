import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_relative_date(date_str):
    today = datetime.today()
    if "Heute" in date_str:
        return today.strftime("%d.%m.%Y")
    elif "Gestern" in date_str:
        return (today - timedelta(days=1)).strftime("%d.%m.%Y")
    return date_str  # Keep the original if it's already a date

def calculate_days_posted(date_str):
    try:
        listing_date = datetime.strptime(date_str, "%d.%m.%Y")
        return (datetime.today() - listing_date).days
    except ValueError:
        return None  # If the format is unexpected

def is_new_bike(title, description):
    keywords = ["neu", "new", "ungebraucht", "brandneu"]
    combined_text = (title + " " + description).lower()
    return any(keyword in combined_text for keyword in keywords)

def extract_price(price_str):
    if not price_str:
        return None
    price_numbers = re.findall(r'\d+', price_str.replace("'", ""))
    return int("".join(price_numbers)) if price_numbers else None

def is_bargain(price):
    return price is not None and price < 3000

def clean_scraped_data(scraped_data):
    try:
        scraped_data['price'] = scraped_data['price'].apply(lambda x: str(x) if pd.notnull(x) else "")
        scraped_data['price'] = scraped_data['price'].apply(lambda x: extract_price(x) if isinstance(x, str) and x else None)

        scraped_data['date'] = scraped_data['date'].apply(lambda x: convert_relative_date(x) if pd.notnull(x) else "No date found")
        scraped_data['date'] = pd.to_datetime(scraped_data['date'], errors='coerce', dayfirst=True)

        scraped_data['days_posted'] = scraped_data['date'].apply(lambda x: calculate_days_posted(x.strftime("%d.%m.%Y")) if pd.notnull(x) else 0)
        
        return scraped_data
    except Exception as e:
        logging.error(f"Error cleaning scraped data: {e}")
        return None

def scrape_tutti_bikes(url, max_pages=2):
    bike_data = []
    page = 1

    while page <= max_pages:
        response = requests.get(f"{url}&page={page}")
        if response.status_code != 200:
            logging.error(f"Error fetching page {page}: {response.status_code}")
            break

        soup = BeautifulSoup(response.text, 'html.parser')
        listings = soup.find_all("div", class_="mui-style-qlw8p1")

        if not listings:
            logging.warning("No more listings found.")
            break

        for listing in listings:
            link_element = listing.find("a", href=True)
            listing_url = f"https://www.tutti.ch{link_element['href']}" if link_element else None

            title_element = listing.find("div", class_="MuiBox-root mui-style-1haxbqe")
            title = title_element.text.strip() if title_element else "No title found"

            desc_element = listing.find("div", class_="MuiBox-root mui-style-xe4gv6")
            description = desc_element.text.strip() if desc_element else "No description found"

            price_element = listing.find("div", class_="MuiBox-root mui-style-1fhgjcy")
            price_span = price_element.find("span") if price_element else None
            price_str = price_span.text.strip() if price_span else None
            price = extract_price(price_str) if price_str else None

            date_place_element = listing.find("span", class_="MuiTypography-root MuiTypography-body1 mui-style-1846fkf")
            date_place = date_place_element.text.strip() if date_place_element else None

            place_with_zip, formatted_date, days_posted = "No place found", "No date found", None

            if date_place:
                date_place_parts = date_place.split(", ")
                if len(date_place_parts) == 3:
                    place = date_place_parts[0]
                    zip_code = date_place_parts[1]
                    date = date_place_parts[2]
                    place_with_zip = f"{place} {zip_code}"
                else:
                    place_with_zip, date = date_place, None
                
                formatted_date = convert_relative_date(date) if date else "No date found"
                days_posted = calculate_days_posted(formatted_date) if formatted_date != "No date found" else None

            is_new = is_new_bike(title, description)

            image_element = listing.find("img")
            image_url = image_element["src"] if image_element else "No image found"

            bike_data.append({
                "url": listing_url,
                "title": title,
                "description": description,
                "price": price,
                "place": place_with_zip,
                "date": formatted_date,
                "days_posted": days_posted,
                "is_new": is_new,
                "is_bargain": is_bargain(price),  # Check if the bike is a bargain
                "image": image_url
            })

        logging.info(f"Scraped page {page} with {len(listings)} listings.")
        page += 1

    df = pd.DataFrame(bike_data)
    df = clean_scraped_data(df)

    return df