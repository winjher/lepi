import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the navigation menu items
MENU_ITEMS = [
    "Dashboard",
    "Tasks",  # Renamed "Task" to "Tasks" to be more intuitive
]

# Define functions to display content for each section
def display_dashboard(dashboardData, st): # Added st as argument
    st.subheader("Dashboard")
    cols = st.columns(2)  # Arrange items in two columns

    with cols[0]:
        st.subheader("Plant Inventory")
        st.write(f"Flowering Plants: {dashboardData['plantInventory']['floweringPlants']}")
        st.write(f"Host Plants: {dashboardData['plantInventory']['hostPlants']}")

    with cols[1]:
        st.subheader("Butterfly Population")
        st.write(f"Total Butterflies: {dashboardData['butterflyPopulation']['totalButterflies']}")
        st.write(f"Eggs: {dashboardData['butterflyPopulation']['eggs']}")
        st.write(f"Pupae: {dashboardData['butterflyPopulation']['pupae']}")

    st.subheader("Care Recommendations")
    for recommendation in dashboardData['careRecommendations']:
        st.write(f"- {recommendation}")

    st.subheader("Analytics")
    st.write(f"Butterfly Emergence Rate: {dashboardData['analytics']['butterflyEmergenceRate']}")
    st.write(f"Egg Hatching Rate: {dashboardData['analytics']['eggHatchingRate']}")
    st.write(f"Average Lifespan: {dashboardData['analytics']['averageLifespan']}")



def display_task(st): #added st as argument
    st.title("🦋 Task Manager")
    st.write("This app provides task information about butterflies and their care.")
    # Define a CSV file name to store the tasks data
    csv_file = "./data/tasks.csv"
    care_csv_file = "./data/care_data.csv"  # Define CSV for care data

    # Define the list of butterfly species
    species_list = [
        "Butterfly-Clippers",
        "Butterfly-Common Jay",
        "Butterfly-Common Lime",
        "Butterfly-Common Mime",
        "Butterfly-Common Mormon",
        "Butterfly-Emerald Swallowtail",
        "Butterfly-Golden Birdwing",
        "Butterfly-Gray Glassy Tiger",
        "Butterfly-Great Eggfly",
        "Butterfly-Great Yellow Mormon",
        "Butterfly-Paper Kite",
        "Butterfly-Pink Rose",
        "Butterfly-Plain Tiger",
        "Butterfly-Red Lacewing",
        "Butterfly-Scarlet Mormon",
        "Butterfly-Tailed Jay",
        "Moth-Atlas",
        "Moth-Giant Silk",
    ]
    # Initialize session state for tasks and care data if not already initialized
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []
    if 'care_data' not in st.session_state:
        st.session_state.care_data = []

    # --- Sidebar Menu ---
    st.sidebar.title("Task Management") # Moved sidebar elements here
    menu_selection = st.sidebar.radio(
        "Choose an action",
        [
            "Register Task",
            "View Tasks",
            "View Task Distribution",
            "Record Care Activity",
            "View Care Activities",
        ],
    )

    # --- Main Content Area ---
    if menu_selection == "Register Task":
        # Task Registration Form
        st.title("Task Registration")

        # Input fields for task registration
        day = st.date_input("Select the Day")
        hour = st.time_input("Select the Hour")
        activity = st.selectbox(
            "Choose Activity Type",
            [
                "Harvesting Eggs",
                "Harvesting Pupae",
                "Feeding Larvae",
                "Butterfly Foraging",
            ],
        )
        details = st.text_input("Additional Details")
        species = st.selectbox("Select Species", species_list)

        # Button for adding a task
        if st.button("Add Task"):
            new_task = {
                "Day": str(day),
                "Hour": str(hour),
                "Activity": activity,
                "Details": details,
                "Species": species,
            }
            # Append the new task to our session state
            st.session_state.tasks.append(new_task)
            # Save the updated tasks list to CSV
            try:
                df = pd.DataFrame(st.session_state.tasks)
                df.to_csv(csv_file, index=False)
                st.success(f"Task added and saved to {csv_file}: {new_task}")
            except Exception as e:
                st.error(f"Error saving task to CSV: {e}. Task added to session, but not saved.")
                st.session_state.tasks.append(new_task)  # Ensure task is added even if CSV save fails

    elif menu_selection == "View Tasks":
        # Display Tasks
        st.title("View Tasks")
        if st.session_state.tasks:
            df_tasks = pd.DataFrame(st.session_state.tasks)
            st.dataframe(df_tasks)  # Display as a DataFrame
        else:
            st.info("No tasks registered yet.")

    elif menu_selection == "View Task Distribution":
        # Display the registered tasks as a table and plot
        st.title("Task Distribution")
        if st.session_state.tasks:
            df = pd.DataFrame(st.session_state.tasks)
            st.dataframe(df)

            # Plotting the distribution of tasks by activity as a bar chart
            st.subheader("Task Activity Distribution")
            try:
                fig, ax = plt.subplots()
                df["Activity"].value_counts().plot(kind='bar', ax=ax)
                ax.set_title("Tasks Distribution by Activity")
                ax.set_xlabel("Activity")
                ax.set_ylabel("Count")
                st.pyplot(fig)

                # Plotting the distribution of tasks by species as a bar chart
                st.subheader("Task Species Distribution")
                fig_species, ax_species = plt.subplots()
                df["Species"].value_counts().plot(kind='bar', ax=ax_species)
                ax_species.set_title("Tasks Distribution by Species")
                ax_species.set_xlabel("Species")
                ax_species.set_ylabel("Count")
                st.pyplot(fig_species)

            except Exception as e:
                st.error(f"Error creating plot: {e}.")
        else:
            st.info("No tasks registered yet.")

    elif menu_selection == "Record Care Activity":
        # Record Care Activity
        st.title("Record Care Activity")

        care_day = st.date_input("Select Care Day")
        care_hour = st.time_input("Select Care Hour")
        care_species = st.selectbox("Select Species", species_list)
        care_activity = st.text_input("Care Activity Description")

        if st.button("Record Care"):
            new_care_data = {
                "Day": str(care_day),
                "Hour": str(care_hour),
                "Species": care_species,
                "Activity": care_activity,
            }
            st.session_state.care_data.append(new_care_data)
            st.success(f"Care activity recorded: {new_care_data}")
            # save care data
            try:
                df_care = pd.DataFrame(st.session_state.care_data)
                df_care.to_csv(care_csv_file, index=False)
                st.success(f"Care activity saved to {care_csv_file}")  # Add success message for saving
            except Exception as e:
                st.error(f"Error saving care data to CSV: {e}")

    elif menu_selection == "View Care Activities":
        # View Care Activities
        st.title("View Care Activities")
        if st.session_state.care_data:
            df_care = pd.DataFrame(st.session_state.care_data)
            st.dataframe(df_care)
        else:
            st.info("No care activities recorded yet.")



# Simulated data (replace with your actual data source)
notifications = [
    {"message": "Daily reminder: Clean cages and refill food plates.", "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
    {"message": "Warning: Potential disease outbreak detected. Monitor butterflies closely.", "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
    {"message": "Notification: Butterfly emergence expected in 2 days.", "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
]

dashboardData = {
    "plantInventory": {
        "floweringPlants": 10,
        "hostPlants": 5,
        "plant alert": "leaves = {0: 'with leaves', 1: 'out leaves'}"

    },
    "butterflyPopulation": {
        "totalButterflies": 25,
        "eggs": 15,
        "pupae": 8,
    },
    "careRecommendations": [
        "Refill food plates today.",
        "Check for signs of disease in butterflies.",
        "Monitor ovipositing stems for eggs."
    ],
    "analytics": {
        "butterflyEmergenceRate": "90%",
        "eggHatchingRate": "85%",
        "averageLifespan": "3 weeks",
    },
}
# Streamlit app
st.title("Butterfly Care Management System")



# Create a dictionary to map menu items to their corresponding functions
MENU_FUNCTIONS = {
    "Dashboard": lambda st: display_dashboard(dashboardData, st),
    "Tasks": display_task,  # Corrected function name
}
# Create the sidebar menu
st.sidebar.title("Navigation")
selected_item = st.sidebar.selectbox("Choose an option", MENU_ITEMS)



# Display the content based on the selected item
if selected_item == "Dashboard":
    display_dashboard(dashboardData, st)
elif selected_item == "Tasks":
    display_task(st)
else:
    st.write(f"Content for '{selected_item}' is not yet implemented.")

def configure_notifications(st):
    st.subheader("Notification Configuration")
    st.write("Here, you can set up how you want to be notified.")

    # Use checkboxes for enabling/disabling notifications
    email_notifications = st.checkbox("Enable Email Notifications", value=True)
    sms_notifications = st.checkbox("Enable SMS Notifications", value=False)
    app_notifications = st.checkbox("Enable App Notifications", value=True)

    # Use sliders for setting notification frequency
    email_frequency = st.slider("Email Frequency (hours)", 1, 24, 6)
    sms_frequency = st.slider("SMS Frequency (hours)", 1, 24, 12)
    app_frequency = st.slider("App Frequency (hours)", 1, 24, 3)

    # Use a multiselect for choosing which events trigger notifications
    notification_events = st.multiselect(
        "Notify me about:",
        [
            "Daily Summary",
            "Disease Outbreak",
            "Emergence Alert",
            "Low Food/Water",
            "Task Reminder"
        ],
        default=["Daily Summary", "Disease Outbreak", "Emergence Alert"]
    )
    if st.button("Save Notification Settings"):
        st.success("Notification settings saved!")
        # In a real application, you would save these settings to a database or config file.
        #  st.write(f"Email: {email_notifications}, Frequency: {email_frequency} hours")
        #  st.write(f"SMS: {sms_notifications}, Frequency: {sms_frequency} hours")
        #  st.write(f"App: {app_notifications}, Frequency: {app_frequency} hours")
        #  st.write(f"Events: {', '.join(notification_events)}")

if st.button("Configure Notifications"):
    configure_notifications(st)

#display_notifications(notifications)
