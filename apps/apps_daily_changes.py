import streamlit as st
import datetime
from datetime import timedelta
import pandas as pd

# Title
st.title("Proper Care Management System for Butterfly Adults")
st.subheader("Foraging")

# Floral Sources
st.header("Floral Sources")
st.write("Provide a variety of flowering plants inside the butterfly house or supplement with fresh petals and sugar syrup.")

# Ant Prevention
st.header("Ant Prevention")
st.write("Use a water-filled plate to prevent ants from accessing the food.")

# Flower Maintenance
st.header("Flower Maintenance")
if st.button("Replace Spoiled Flowers"):
    st.success("Spoiled flowers have been replaced!")

# Initialize daily change schedule
if 'daily_changes' not in st.session_state:
    st.session_state['daily_changes'] = []

# Add daily change to the schedule
def add_daily_change():
    message = "Today's change: Refresh the sugar syrup."
    date = datetime.date.today()
    st.session_state['daily_changes'].append({'Date': date, 'Message': message})

# Schedule today's daily change
add_daily_change()

# Custom Change Scheduling
st.header("Schedule Custom Change")
custom_date = st.date_input("Select Date for Custom Change", min_value=datetime.date.today())
custom_message = st.text_input("Enter Custom Change Message")

if st.button("Schedule Custom Change"):
    if custom_message.strip() == "":
        st.error("Please enter a valid message for the custom change.")
    else:
        st.session_state['daily_changes'].append({'Date': custom_date, 'Message': custom_message})
        st.success("Custom change scheduled successfully!")

# Display Daily Changes
st.header("Scheduled Changes")
if st.session_state['daily_changes']:
    changes_df = pd.DataFrame(st.session_state['daily_changes'])
    st.table(changes_df)

    # Save to CSV
    if st.button("Save Schedule to CSV"):
        changes_df.to_csv("C:/Users/jerwin/Documents/GitHub/bilabila/Data/daily_changes_schedule.csv", index=False)
        st.success("Schedule saved to 'C:/Users/jerwin/Documents/GitHub/bilabila/Data/daily_changes_schedule.csv'")
else:
    st.write("No changes scheduled yet.")

