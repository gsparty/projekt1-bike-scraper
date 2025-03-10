import pandas as pd
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["tutti_bikes"]
collection = db["listings"]


# Fetch data
data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB's _id field
if not data:
    print("⚠️ No data found in MongoDB!")
else:
    print("✅ Sample data from MongoDB:", data[:3])  # Print first 3 entries

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("bike_listings.csv", index=False)
print("✅ Dataset exported as bike_listings.csv")
