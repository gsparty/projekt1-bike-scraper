import streamlit as st
import pandas as pd
from scraper import scrape_tutti_bikes, save_to_mongo
from model import prepare_data, extract_features, load_historical_data, predict_days_until_sold, train_regression_model, add_days_until_sold
import requests
import joblib

# Constants
BASE_URL = "https://www.tutti.ch/de/q/motorraeder/Ak8CrbW90b3JjeWNsZXOUwMDAwA?sorting=newest"
HISTORICAL_DATA_PATH = 'historical_data.csv'  # Update path

# Debugging function
def check_ip():
    """Displays the public IP address."""
    try:
        response = requests.get("https://api64.ipify.org?format=json")
        st.sidebar.write(f"🌍 Your Public IP: {response.json()['ip']}")  # Moved to sidebar
    except Exception as e:
        st.sidebar.error(f"❌ Failed to fetch public IP: {e}")  # Moved to sidebar

# Streamlit UI
def main():
    st.title("Bike Scraper 🚴‍♂️ Brodydan")

    # Sidebar for additional options
    st.sidebar.title("Options")
    if st.sidebar.button("Check IP"):
        check_ip()

    # Load historical data
    try:
        historical_data = load_historical_data(HISTORICAL_DATA_PATH)
        if historical_data is not None:
            historical_data = add_days_until_sold(historical_data)  # Add simulated column
            historical_data_df = prepare_data(historical_data)
            historical_data_df = extract_features(historical_data_df)
            st.sidebar.success("✅ Historical data loaded and prepared successfully.")
        else:
            st.sidebar.error("❌ Failed to load historical data.")
            return
    except Exception as e:
        st.sidebar.error(f"❌ Error loading historical data: {e}")
        return

    # Main app functionality
    st.header("Scrape Listings")
    if st.button("Start Scraping"):
        with st.spinner("Scraping in progress..."):
            data = scrape_tutti_bikes(BASE_URL, max_pages=5)
            st.write(f"Scraped {len(data)} listings.")

        if not data:  # Check if the list is empty
            st.warning("❌ No listings found. Check the website or try later.")
        else:
            st.success(f"✅ Scraped {len(data)} listings successfully!")
            scraped_df = pd.DataFrame(data)  # Convert to DataFrame

            # Save scraped data to MongoDB
            save_to_mongo(data)

            # Store scraped data in session state to persist across reruns
            st.session_state.scraped_df = scraped_df

    # Display scraped listings and dropdown if data exists
    if 'scraped_df' in st.session_state:
        scraped_df = st.session_state.scraped_df

        # Display all scraped listings in a table
        st.write("All Scraped Listings:")
        st.dataframe(scraped_df)  # Show all scraped listings in a table

        # Add a dropdown to select a listing
        selected_index = st.selectbox(
            "Select a listing to predict time to sell:",
            range(len(scraped_df)),
            format_func=lambda x: f"Listing {x + 1}: {scraped_df.iloc[x]['title']}"
        )

        # Display selected listing details
        selected_listing = scraped_df.iloc[selected_index].to_dict()
        st.write("Selected Listing Details:")
        st.write(selected_listing)

        # Predict days_until_sold for the selected listing
        if st.button("Predict Time to Sell"):
            try:
                # Load the regression model
                regression_model = joblib.load('regression_model.pkl')

                # Ensure the selected listing has the required columns
                if 'location' not in selected_listing:
                    selected_listing['location'] = 'unknown'  # Add default value if missing
                if 'days_posted' not in selected_listing:
                    selected_listing['days_posted'] = 0  # Add default value if missing

                # Predict days_until_sold and confidence
                days_until_sold, confidence = predict_days_until_sold(regression_model, selected_listing)

                if days_until_sold is not None:
                    st.success(f"🕒 Predicted Time to Sell: {days_until_sold:.1f} days")
                    st.success(f"🎯 Confidence: {confidence:.1f}%")
                else:
                    st.error("❌ Failed to predict time to sell.")
            except Exception as e:
                st.error(f"❌ Error predicting time to sell: {e}")

    # Train regression model section
    st.sidebar.header("Train Regression Model")
    if st.sidebar.button("Train Regression Model"):
        with st.spinner("Training regression model..."):
            regression_model = train_regression_model(historical_data_df)
            if regression_model:
                st.sidebar.success("✅ Regression model trained successfully!")
            else:
                st.sidebar.error("❌ Regression model training failed.")

if __name__ == "__main__":
    main()