import requests

url = "https://www.tutti.ch/de/q/motorraeder/Ak8CrbW90b3JjeWNsZXOUwMDAwA?sorting=newest"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Accept-Language": "de-DE,de;q=0.9",
    "Referer": "https://www.google.com",
    "DNT": "1",
    "Connection": "keep-alive"
}

response = requests.get(url, headers=headers)

print(f"Status Code: {response.status_code}")
print(f"Response Content: {response.text[:500]}")

response = requests.get("https://api64.ipify.org?format=json")
print(response.json())  # Show your public IP

