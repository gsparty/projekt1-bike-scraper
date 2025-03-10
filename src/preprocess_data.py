import pandas as pd
from pymongo import MongoClient
import re

# Constants
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "tutti_bikes"
COLLECTION_NAME = "listings"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Fetch data from MongoDB
data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB ID
df = pd.DataFrame(data)

# Function to clean price and convert to number
def clean_price(price):
    if isinstance(price, str):
        price = re.sub(r"[^\d]", "", price)  # Remove non-numeric characters
        return int(price) if price else None
    return None

# Function to extract city from "date_place"
def extract_city(date_place):
    if isinstance(date_place, str):
        parts = date_place.split(",")  # Assume city is last part after a comma
        return parts[-1].strip() if len(parts) > 1 else "Unknown"
    return "Unknown"

# Apply transformations
df["price"] = df["price"].apply(clean_price)
df["city"] = df["date_place"].apply(extract_city)
df["title_length"] = df["title"].apply(lambda x: len(x) if isinstance(x, str) else 0)
df["desc_length"] = df["description"].apply(lambda x: len(x) if isinstance(x, str) else 0)

# Handle missing prices (fill with median)
df["price"].fillna(df["price"].median(), inplace=True)

# Create a binary "high_price" target variable (above median price = 1)
price_median = df["price"].median()
df["high_price"] = (df["price"] > price_median).astype(int)

# Save cleaned data
df.to_csv("preprocessed_data.csv", index=False)
print("✅ Data Preprocessed & Saved as preprocessed_data.csv!")
