import streamlit as st
import pandas as pd
import datetime
import os

# --- Configuration ---
CSV_FILE = 'login_records.csv'
VALID_USERS = {
    "user1": "pass1",
    "admin": "adminpass"
}

# IMPORTANT: This MUST be the actual, full URL where your Lepidoptera.py Streamlit app is deployed
BUTTERFLY_APP_URL = "https://winjher.github.io/jerapp/home.html" # Keeping your example URL

# --- Functions for CSV handling ---
def initialize_csv():
    """Creates the CSV file with headers if it doesn't exist."""
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=["timestamp", "username", "status"])
        df.to_csv(CSV_FILE, index=False)

def record_login_attempt(username, status):
    """Records a login attempt to the CSV file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_record = pd.DataFrame([{
        "timestamp": timestamp,
        "username": username,
        "status": status
    }])
    new_record.to_csv(CSV_FILE, mode='a', header=False, index=False)

def get_login_records():
    """Reads all login records from the CSV file."""
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    return pd.DataFrame(columns=["timestamp", "username", "status"])

# --- Streamlit App ---
st.set_page_config(page_title="Streamlit Login App", layout="centered")
st.title("Login")

initialize_csv()

# --- AUTOMATIC LOGIN MODIFICATION ---
# Set logged_in to True and provide a default username for automatic login
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.session_state['logged_in'] = True
    st.session_state['username'] = "auto_user" # You can set any default username here
    # Record this as an automatic login
    record_login_attempt("auto_user", "AUTO_SUCCESS")
    #st.experimental_rerun() # Rerun to display the logged-in content or redirect

# Display content after "login"
st.success(f"Automatically logged in as {st.session_state.get('username', 'User')}.")
st.subheader("Welcome to the Protected Area!")
st.write("This content is accessible because of automatic login.")

# --- REDIRECTION LOGIC ---
if BUTTERFLY_APP_URL:
    st.write(f"Redirecting you to the [Butterfly Apps]({BUTTERFLY_APP_URL}) page...")
    # Use st.markdown with HTML and JavaScript for immediate redirection
    st.markdown(f"""
        <script>
            window.location.href = "{BUTTERFLY_APP_URL}";
        </script>
        """, unsafe_allow_html=True)
else:
    st.warning("BUTTERFLY_APP_URL is not set. Cannot redirect automatically.")

st.write("---")

st.subheader("Login Records")
records_df = get_login_records()
if not records_df.empty:
    st.dataframe(records_df)

    csv_data = records_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Login Records CSV",
        data=csv_data,
        file_name="login_records.csv",
        mime="text/csv",
        key="download_records_csv"
    )
else:
    st.info("No login records yet.")

# A logout button still makes sense for demonstration or to reset
if st.button("Logout"):
    st.session_state['logged_in'] = False
    if 'username' in st.session_state:
        del st.session_state['username']
    #st.experimental_rerun() # Rerun to reset, although auto-login will re-engage