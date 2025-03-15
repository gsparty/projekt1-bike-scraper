from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"  # Change if necessary
DB_NAME = "projekt1"
COLLECTION_NAME = "bike_listings"

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Try inserting a test document
    test_doc = {"test": "connection_successful"}
    collection.insert_one(test_doc)
    
    # Verify insertion
    result = collection.find_one({"test": "connection_successful"})
    if result:
        print("✅ MongoDB connection successful and data inserted!")
    else:
        print("❌ Data insertion failed.")
    
    client.close()
except Exception as e:
    print(f"❌ MongoDB connection error: {e}")
