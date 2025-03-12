import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re

def convert_relative_date(date_str):
    """Converts relative dates like 'Heute' or 'Gestern' to actual dates."""
    today = datetime.today()
    if "Heute" in date_str:
        return today.strftime("%d/%m/%Y")
    elif "Gestern" in date_str:
        return (today - timedelta(days=1)).strftime("%d/%m/%Y")
    return date_str  # Keep the original if it's already a date

def calculate_days_posted(date_str):
    """Calculates how many days ago the bike was posted."""
    try:
        listing_date = datetime.strptime(date_str, "%d/%m/%Y")
        return (datetime.today() - listing_date).days
    except ValueError:
        return None  # If the format is unexpected

def is_new_bike(title, description):
    """Determines if the listing describes a new bike."""
    keywords = ["neu", "new", "ungebraucht", "brandneu"]
    combined_text = (title + " " + description).lower()
    return any(keyword in combined_text for keyword in keywords)

def extract_price(price_str):
    """Extracts numeric price from a formatted string."""
    if not price_str:
        return None
    price_numbers = re.findall(r'\d+', price_str.replace("'", ""))  # Handle CHF 1'200.– format
    return int("".join(price_numbers)) if price_numbers else None

def scrape_tutti_bikes(url, max_pages=2):
    """Scrapes bike listings from Tutti and returns structured data."""
    bike_data = []
    page = 1

    while page <= max_pages:
        response = requests.get(f"{url}&page={page}")
        if response.status_code != 200:
            print(f"Error fetching page {page}: {response.status_code}")
            break

        soup = BeautifulSoup(response.text, 'html.parser')
        listings = soup.find_all("div", class_="mui-style-qlw8p1")  # Update class if necessary

        if not listings:
            print("No more listings found.")
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

            date_place_element = listing.find("span", class_="MuiTypography-root MuiTypography-body1 mui-style-18rb2ut")
            date_place = date_place_element.text.strip() if date_place_element else None

            if date_place:
                date_place_parts = date_place.split(", ")
                place_with_zip = date_place_parts[0] if len(date_place_parts) > 0 else "No place found"
                date = ", ".join(date_place_parts[1:]) if len(date_place_parts) > 1 else "No date found"
                formatted_date = convert_relative_date(date) if date else None
                days_posted = calculate_days_posted(formatted_date) if formatted_date else None
            else:
                place_with_zip, formatted_date, days_posted = "No place found", "No date found", None

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
                "is_bargain": None,  # Placeholder
                "image": image_url
            })

        print(f"Scraped page {page} with {len(listings)} listings.")
        page += 1

    return bike_data
