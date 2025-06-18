import streamlit as st
import pandas as pd
import os
import csv
from datetime import datetime

# --- Configuration ---
CSV_FILE = 'purchasers.csv'

# --- Company Options ---
COMPANIES =  [
    "Butterfly World Inc.", "Winged Wonders Ltd.", "Nature's Flight Co.",
    "Moth & More", "Flutterby Farms", "Papilio Partners", "Other"
]

# --- Species Options ---
ALL_SPECIES = [
    'Butterfly-Clippers', 'Butterfly-Common Jay', 'Butterfly-Common Lime',
    'Butterfly-Common Mime', 'Butterfly-Common Mormon', 'Butterfly-Emerald Swallowtail',
    'Butterfly-Golden Birdwing', 'Butterfly-Great Eggfly', 'Butterfly-Great Yellow Mormon',
    'Butterfly-Grey Glassy Tiger', 'Butterfly-Paper Kite', 'Butterfly-Pink Rose',
    'Butterfly-Plain Tiger', 'Butterfly-Red Lacewing', 'Butterfly-Scarlet Mormon',
    'Butterfly-Tailed Jay', 'Moth-Atlas', 'Moth-GiantSilk'
]


# --- CSV Handling ---
def initialize_csv():
    """Ensures the CSV file exists with the correct headers."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow()

def save_sale_to_csv(buyer, quantity, species, company):
    """Saves a single sale record to the CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Prepare data as a list to match CSV row format
    row_data = [timestamp, buyer, company, quantity, species]
    with open(CSV_FILE, 'a', newline='') as f: # Use 'a' for append mode
        writer = csv.writer(f)
        writer.writerow(row_data)

def load_sales_from_csv():
    """Loads all sale records from the CSV file."""
    if os.path.exists(CSV_FILE):
        try:
            return pd.read_csv(CSV_FILE)
        except pd.errors.ParserError as e:
            st.error(f"Error reading CSV file: {e}. Please ensure the '{CSV_FILE}' file is correctly formatted. You might need to delete it to start fresh.")
            # Return an empty DataFrame with correct columns to prevent further errors
            return pd.DataFrame(columns=['Timestamp','Purchaser Name','Company','Quantity','Species'])
    # Return an empty DataFrame with correct columns if file doesn't exist
    return pd.DataFrame(columns=['Timestamp','Purchaser Name','Company','Quantity','Species'])

# --- Streamlit App ---
def main():
    # Set Streamlit page configuration for better appearance
    st.set_page_config(page_title="Company Butterfly Pupae Sales Tracker", layout="centered")
    st.title("🦋 Company Butterfly Pupae Sales Tracker")

    initialize_csv() # Ensure the CSV file exists with headers

    st.header("Enter Buyer Information")
    buyer_name = st.text_input("Purchaser/Buyer Name:")

    company_selection = st.selectbox("Select Company:", COMPANIES)
    company_to_save = company_selection # Default to selected company

    if company_selection == "Other":
        other_company_name = st.text_input("Enter Company Name:")
        # Use the entered name if provided, otherwise a placeholder for "Other"
        company_to_save = other_company_name if other_company_name else "Other (Not Specified)"

    st.header("Pupae Details")
    pupae_quantity = st.number_input("Quantity of Pupae:", min_value=1, value=10)
    pupae_species = st.selectbox("Species:", ALL_SPECIES)

    st.write("---")  # Separator

    if st.button("Record Sale"):
        # Input validation
        if not buyer_name:
            st.warning("Please enter the **Purchaser/Buyer Name**.")
        elif company_selection == "Other" and not other_company_name:
            st.warning("Please enter the **Company Name** for 'Other' selection.")
        else:
            # Save the sale record
            save_sale_to_csv(buyer_name, pupae_quantity, pupae_species, company_to_save)
            st.success(f"Sale recorded for: **{buyer_name}** from **{company_to_save}**!")
            st.rerun() # Rerun the app to clear inputs and refresh the sales history

    st.write("---")
    st.subheader("All Recorded Sales")
    sales_df = load_sales_from_csv()
    if not sales_df.empty:
        # Display the dataframe, sorted by Timestamp in descending order (most recent first)
        st.dataframe(sales_df.sort_values(by='Timestamp', ascending=False))
    else:
        st.info("No sales recorded yet. Start by entering a new sale above!")

if __name__ == "__main__":
    main()
