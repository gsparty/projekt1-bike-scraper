from model import (
    create_model, 
    predict_selling_probability, 
    scrape_tutti_bikes, 
    prepare_data, 
    extract_features, 
    visualize_trends
)
import datetime

if __name__ == "__main__":
    # Step 1: Scrape data
    bike_data = scrape_tutti_bikes("https://www.tutti.ch/de/q/motorraeder/Ak8CrbW90b3JjeWNsZXOUwMDAwA?sorting=newest&page=1")

    if not bike_data:
        print("No data scraped. Exiting program.")
        exit()

    # Debugging log to inspect scraped data
    print("Sample of scraped data:", bike_data[:5])  # Print first 5 items to check

    # Step 2: Prepare data
    df = prepare_data(bike_data)
    print("Prepared dataframe:")
    print(df.head())  # Check the dataframe structure

    # Step 3: Extract additional features
    df = extract_features(df)

    # Step 4: Visualize trends
    visualize_trends(df)

    # Step 5: Create model
    model = create_model(df)
    
    # Step 6: Test prediction with sample data
    test_price = 500
    test_days_posted = 20
    sale_chance = predict_selling_probability(model, test_price, test_days_posted)
    
    print(f"Prediction: {'Sold' if sale_chance > 0.5 else 'Not Sold'} with a confidence of {sale_chance:.2%}")
