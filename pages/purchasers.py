import streamlit as st
import pandas as pd
import os
import sqlite3
from datetime import datetime
from bcrypt import hashpw, checkpw, gensalt # For password hashing

# --- Configuration ---
DATABASE_FILE = 'users.db' # SQLite database for user accounts
SALES_CSV_FILE = 'sales_records.csv' # CSV for sales data (renamed for clarity)

# --- Database Functions for User Management ---
def init_db():
    """Initializes the SQLite database and creates the users table if it doesn't exist."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username, password):
    """Adds a new user to the database after hashing the password."""
    hashed_password = hashpw(password.encode('utf-8'), gensalt()).decode('utf-8')
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        st.error("Username already exists!")
        return False
    finally:
        conn.close()

def verify_user(username, password):
    """Verifies a user's password against the hashed password in the database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()

    if result:
        stored_hashed_password = result[0].encode('utf-8')
        return checkpw(password.encode('utf-8'), stored_hashed_password)
    return False

# --- CSV Functions for Sales Data (Modified for multi-user) ---
def save_sale_to_csv(breeder, buyer, quantity, species):
    """Saves a single sale record to the CSV file, including the breeder."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_record = {
        'Timestamp': timestamp,
        'Breeder/Seller': breeder, # New column for the breeder/seller
        'Purchaser Name': buyer,
        'Quantity': quantity,
        'Species': species
    }
    new_df = pd.DataFrame([new_record])

    if not os.path.exists(SALES_CSV_FILE):
        new_df.to_csv(SALES_CSV_FILE, index=False)
    else:
        new_df.to_csv(SALES_CSV_FILE, mode='a', header=False, index=False)

def load_sales_from_csv(breeder=None):
    """
    Loads all sale records from the CSV file.
    If a breeder is provided, filters the sales for that specific breeder.
    """
    if os.path.exists(SALES_CSV_FILE):
        df = pd.read_csv(SALES_CSV_FILE)
        if breeder:
            return df[df['Breeder/Seller'] == breeder]
        return df # Return all if no breeder specified (e.g., for admin view)
    return pd.DataFrame(columns=['Timestamp', 'Breeder/Seller', 'Purchaser Name', 'Quantity', 'Species'])

# --- Streamlit App Layout ---
def login_page():
    """Displays the login and signup forms."""
    st.sidebar.subheader("Login / Sign Up")
    choice = st.sidebar.radio("Go to", ["Login", "Sign Up"])

    if choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if verify_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.sidebar.success(f"Welcome, {username}!")
                st.rerun() # Rerun to switch to the main app
            else:
                st.sidebar.error("Invalid Username or Password")

    elif choice == "Sign Up":
        new_username = st.sidebar.text_input("New Username")
        new_password = st.sidebar.text_input("New Password", type="password")
        confirm_password = st.sidebar.text_input("Confirm Password", type="password")

        if st.sidebar.button("Sign Up"):
            if new_password == confirm_password:
                if add_user(new_username, new_password):
                    st.sidebar.success("Account created! Please login.")
                else:
                    # Error handled by add_user already
                    pass
            else:
                st.sidebar.error("Passwords do not match.")

def main_app():
    """The main application logic after a user is logged in."""
    st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        del st.session_state.username
        st.rerun() # Rerun to go back to login page

    st.title(f"Pupae Sales Tracker for {st.session_state.username}")

    st.header("Enter Sale Information")

    buyer_name = st.text_input("Purchaser/Buyer Name:")

    all_species = [
        'Butterfly-Clippers', 'Butterfly-Common Jay', 'Butterfly-Common Lime',
        'Butterfly-Common Mime', 'Butterfly-Common Mormon', 'Butterfly-Emerald Swallowtail',
        'Butterfly-Golden Birdwing', 'Butterfly-Great Eggfly', 'Butterfly-Great Yellow Mormon',
        'Butterfly-Grey Glassy Tiger', 'Butterfly-Paper Kite', 'Butterfly-Pink Rose',
        'Butterfly-Plain Tiger', 'Butterfly-Red Lacewing', 'Butterfly-Scarlet Mormon',
        'Butterfly-Tailed Jay', 'Moth_Atlas', 'Moth-GiantSilk',
    ]

    pupae_quantity = st.number_input("Quantity of Pupae:", min_value=1, value=10)
    pupae_species = st.selectbox("Species:", all_species)

    st.write("---")

    if st.button("Record Sale"):
        if buyer_name:
            save_sale_to_csv(st.session_state.username, buyer_name, pupae_quantity, pupae_species)
            st.success(f"Sale Recorded for: **{buyer_name}** and saved!")
            st.rerun() # Refresh the displayed sales
        else:
            st.warning("Please enter the **Purchaser/Buyer Name** before recording the sale.")

    st.write("---")
    st.subheader(f"Your Recent Sales ({st.session_state.username})")

    sales_df = load_sales_from_csv(st.session_state.username) # Load only current user's sales
    if not sales_df.empty:
        st.dataframe(sales_df.sort_values(by='Timestamp', ascending=False))
    else:
        st.info("No sales recorded yet. Start by entering a new sale above!")

def main():
    """Main entry point of the Streamlit application."""
    init_db() # Ensure database is initialized

    # Initialize session state for login if not already present
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main()