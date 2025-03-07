import requests
from bs4 import BeautifulSoup
from datetime import datetime

def scrape_tutti_bikes(url):
    """Scrapes bike listings from Tutti and returns structured data."""
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Fehler beim Abruf der Seite: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    
    bike_data = []
    listings = soup.find_all("div", class_="listing")  # Prüfe die echte Klassenbezeichnung auf Tutti
    
    for listing in listings:
        title = listing.find("h2").text.strip() if listing.find("h2") else "Kein Titel"
        price = listing.find("span", class_="price").text.strip() if listing.find("span", class_="price") else "Preis unbekannt"
        location = listing.find("span", class_="location").text.strip() if listing.find("span", class_="location") else "Standort unbekannt"
        date_posted = listing.find("time").text.strip() if listing.find("time") else "Kein Datum"
        
        # Convert the date to a standard format (if possible)
        try:
            date_posted = datetime.strptime(date_posted, '%d.%m.%Y')  # Change this format based on the actual format from Tutti
        except ValueError:
            date_posted = None

        bike_data.append({
            "title": title,
            "price": price,
            "location": location,
            "date_posted": date_posted
        })
    
    return bike_data

# Testen
if __name__ == "__main__":
    test_url = "https://www.tutti.ch/de/li/ganze-schweiz/velo"  # Beispiel-URL für Fahrräder
    bikes = scrape_tutti_bikes(test_url)
    for bike in bikes[:5]:  # Zeige die ersten 5 Ergebnisse
        print(bike)
