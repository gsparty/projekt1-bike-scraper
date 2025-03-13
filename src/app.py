import streamlit as st
from scraper import scrape_tutti_bikes
from model import prepare_data, extract_features, determine_bargains, load_historical_data, train_model, load_model
import requests

# Constants
BASE_URL = "https://www.tutti.ch/de/q/motorraeder/Ak8CrbW90b3JjeWNsZXOUwMDAwA?sorting=newest"
HISTORICAL_DATA_PATH = 'historical_data.csv'  # Update path

# Debugging function
def check_ip():
    """Displays the public IP address."""
    try:
        response = requests.get("https://api64.ipify.org?format=json")
        st.write(f"🌍 Your Public IP: {response.json()['ip']}")
    except Exception as e:
        st.error(f"❌ Failed to fetch public IP: {e}")

# Streamlit UI
def main():
    st.title("Bike Scraper 🚴‍♂️")

    if st.button("Check IP"):
        check_ip()

    # Load historical data
    try:
        historical_data = load_historical_data(HISTORICAL_DATA_PATH)
        if historical_data is not None:
            historical_data_df = prepare_data(historical_data)
            historical_data_df = extract_features(historical_data_df)
            st.success("✅ Historical data loaded and prepared successfully.")
        else:
            st.error("❌ Failed to load historical data.")
            return
    except Exception as e:
        st.error(f"❌ Error loading historical data: {e}")
        return

    if st.button("Start Scraping"):
        with st.spinner("Scraping in progress..."):
            data = scrape_tutti_bikes(BASE_URL, max_pages=5)
            st.write(f"Scraped {len(data)} listings.")

        if data.empty:  # Check if the DataFrame is empty
            st.warning("❌ No listings found. Check the website or try later.")
        else:
            st.success(f"✅ Scraped {len(data)} listings successfully!")
            st.write(data.head())  # Show first few results

            # Prepare and visualize data
            try:
                df = prepare_data(data)
                if df is not None:
                    df = extract_features(df)
                    st.write(df.head())  # Show first few rows of the dataframe
                else:
                    st.error("❌ Data preparation failed.")
                    return

                # Determine bargains in the new listings based on historical data
                try:
                    new_listings_df_with_bargains = determine_bargains(df, historical_data_df)
                    st.write(new_listings_df_with_bargains.head())  # Show first few rows with bargains flagged
                except Exception as e:
                    st.error(f"❌ Failed to determine bargains: {e}")
            except Exception as e:
                st.error(f"❌ Error preparing and extracting features: {e}")

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            model = train_model(historical_data_df)
            if model:
                st.success("✅ Model trained successfully!")
            else:
                st.error("❌ Model training failed.")

if __name__ == "__main__":
    main()