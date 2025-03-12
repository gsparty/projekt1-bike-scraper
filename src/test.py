import requests
from bs4 import BeautifulSoup

# Function to fetch the HTML of a page
def fetch_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    try:
        # Sending GET request to the URL
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Will raise an exception for HTTP error responses
        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the HTML: {e}")
        return None

# Example URL (you can replace this with the actual URL you want to scrape)
url = "https://www.tutti.ch/de/market/motorrad"

# Fetch and print the HTML content
html_soup = fetch_html(url)
if html_soup:
    print(html_soup.prettify())  # Pretty print the HTML for easier reading
