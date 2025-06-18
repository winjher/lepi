import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report # Keep if you plan to add evaluation sections later

# Set page configuration at the very beginning
st.set_page_config(layout="wide", page_title="Butterfly App", page_icon="🦋")

# --- Function to set background image ---
def set_background_image(image_path):
    """
    Sets a background image for the Streamlit application using CSS.

    Args:
        image_path (str): The path to the local image file.
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{_get_base64_image(image_path)}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def _get_base64_image(image_path):
    """Helper to convert local image to base64 for CSS background."""
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- Apply the background image ---
# Ensure 'icon/bgbutterfly.jpg' is in the correct path relative to your script
# You might need to create the 'icon' folder and place the image there.
try:
    set_background_image('icon/bg.jpg')
except FileNotFoundError:
    st.warning("Background image 'icon/bg.jpg' not found. Please ensure it's in the correct path.")

# --- Define Paths ---

# !!! FIX: Define MODEL_DIR here !!!
# IMPORTANT: Adjust this path to where your .h5 model files are stored.
# For example: './models/', '../my_ml_models/', or an absolute path


IMAGE_DIR = './butterfly_photos/butterfly/' # Base directory for your images (for displaying species names)
DATA_DIR = './Data/' # Directory for your CSV data files
UPLOAD_DIR = './upload/' # Directory for uploaded images

# Create directories if they don't exist

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True) # Ensure upload directory exists

# Define CSV file names for task management and care activities
CSV_FILE_TASKS = os.path.join(DATA_DIR, "tasks.csv")
CSV_FILE_CARE = os.path.join(DATA_DIR, "care_data.csv")
CSV_FILE_BUTTERFLY_DATA = os.path.join(DATA_DIR, "butterfly_data.csv") # Used for species image counts
CSV_FILE_STAGES_COUNT = os.path.join(DATA_DIR, "stages_records.csv")
CSV_FILE_BUTTERFLY_COUNT = os.path.join(DATA_DIR, "butterfly_count.csv") # Path to butterfly_count.csv
CSV_FILE_BUTTERFLY_TABLE = os.path.join(DATA_DIR, "butterfly_table_species.csv")
CSV_FILE_PUPAE_DEFECTS = os.path.join(DATA_DIR, "pupae_defects_quality_info.csv")
CSV_FILE_LARVAL_DISEASE = os.path.join(DATA_DIR, "larval_diseases_quality_records.csv")
CSV_FILE_BUTTERFLY_SPECIES_DETECTED = os.path.join(DATA_DIR, "Butterfly Species.csv")

# Load initial dataframes (wrap with @st.cache_data for efficiency)
@st.cache_data
def load_csv_data(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            return pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            st.warning(f"'{os.path.basename(file_path)}' is empty. Returning empty DataFrame.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading '{os.path.basename(file_path)}': {e}")
            return pd.DataFrame()
    st.warning(f"'{os.path.basename(file_path)}' not found or empty.")
    return pd.DataFrame()

butterfly_count = load_csv_data(CSV_FILE_BUTTERFLY_COUNT)
butterfly_table = load_csv_data(CSV_FILE_BUTTERFLY_TABLE)
butterfly_stages = load_csv_data(CSV_FILE_STAGES_COUNT)
butterfly_pupae = load_csv_data(CSV_FILE_PUPAE_DEFECTS)
butterfly_larval_disease = load_csv_data(CSV_FILE_LARVAL_DISEASE)
butterfly_species_detected = load_csv_data(CSV_FILE_BUTTERFLY_SPECIES_DETECTED)

# Define class names for models (ensure these match your model's output)
butterfly_species_names = [
    "Butterfly-Clippers", "Butterfly-Common Jay", "Butterfly-Common Lime",
    "Butterfly-Common Mime", "Butterfly-Common Mormon", "Butterfly-Emerald Swallowtail",
    "Butterfly-Golden Birdwing", "Butterfly-Gray Glassy Tiger", "Butterfly-Great Eggfly",
    "Butterfly-Great Yellow Mormon", "Butterfly-Paper Kite", "Butterfly-Pink Rose",
    "Butterfly-Plain Tiger", "Butterfly-Red Lacewing", "Butterfly-Scarlet Mormon",
    "Butterfly-Tailed Jay", "Moth-Atlas", "Moth-Giant Silk",
]

pupaedefects_names = [
    'Ant bites', 'Deformed body', 'Healthy Pupae', 'Old Pupa', 'Overbend', 'Stretch abdomen'
]

larvaldiseases_names = [
    "Anaphylaxis Infection", "Gnathostomiasis", "Healthy", "Nucleopolyhedrosis"
]

lifestages_names = ['Butterfly', 'Eggs', 'Larvae', 'Pupae']

# --- Hostplants Information (Unified) ---
hostplants_info = {
    "Butterfly-Clippers": "Wild Cucumber.",
    "Butterfly-Common Jay": "'Avocado Tree', 'Soursop', 'Sugar Apple', 'Amuyon', 'Indian Tree'.",
    "Butterfly-Common Lime": "'Limeberry', 'Calamondin', 'Pomelo', 'Sweet Orange', 'Calamansi'.",
    "Butterfly-Common Mime": "'Clover Cinnamon', 'Wild Cinnamon'.",
    "Butterfly-Common Mormon": "'Limeberry', 'Calamondin', 'Pomelo', 'Sweet Orange', 'Calamansi', 'Lemoncito'.",
    "Butterfly-Emerald Swallowtail": "'Curry Leafs', 'Pink Lime-Berry Tree'.",
    "Butterfly-Golden Birdwing": "'Dutchman Pipe', 'Indian Birthwort'.",
    "Butterfly-Gray Glassy Tiger": "'Limeberry', 'Calamondin', 'Pomelo', 'Sweet Orange', 'Calamansi'.",
    "Butterfly-Great Eggfly": "Dutchman Pipe', 'Indian Birthwort'.",
    "Butterfly-Great Yellow Mormon": "'Sweet Potato', 'Water Spinach'.",
    "Butterfly-Paper Kite": "'Common Skillpod'.",
    "Butterfly-Pink Rose": "'Dutchman Pipe', 'Indian Birthwort'.",
    "Butterfly-Plain Tiger": "'Crown flower', 'Giant Milkweed'.",
    "Butterfly-Red Lacewing": "'Wild Bush Passion Fruits'.",
    "Butterfly-Scarlet Mormon": "'Calamondin', 'Pomelo', 'Sweet Orange', 'Calamansi'.",
    "Butterfly-Tailed Jay": "'Avocado Tree', 'Soursop', 'Sugar Apple', 'Amuyon', 'Indian Tree'.",
    "Moth-Atlas": "'Amuyon','Gmelina Tree', 'Soursop'.",
    "Moth-Giant Silk": "'Curry Leafs'."
}


# --- Load image counts for species from CSV ---
@st.cache_data(show_spinner=False)
def load_species_image_counts(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.set_index('Species')['Number of Images'].to_dict()
    except FileNotFoundError:
        st.error(f"Species image count data not found at: {file_path}. Using dummy counts.")
        return {} # Return empty dict if file not found
    except pd.errors.EmptyDataError:
        st.warning(f"Species image count CSV at {file_path} is empty. Returning empty dict.")
        return {}
    except Exception as e:
        st.error(f"Error loading species image counts: {e}. Returning empty dict.")
        return {}

species_image_counts = load_species_image_counts(CSV_FILE_BUTTERFLY_DATA)

# --- Generic Prediction Function for any Classifier ---
def classify_image(image_file, model_obj, class_names, img_size=(180, 180)):
    if model_obj is None:
        return "Model Not Loaded", 0.0

    img = Image.open(image_file).resize(img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch dimension
    img_array = img_array / 255.0 # Normalize image data if your model expects it

    predictions = model_obj.predict(img_array, verbose=0) # Added verbose=0 to suppress Keras output

    # Assuming a classification model where the last layer is dense and not necessarily softmax activated
    # Apply softmax to get probabilities
    score = tf.nn.softmax(predictions[0]).numpy() # Convert EagerTensor to NumPy array
    predicted_class_index = np.argmax(score)

    if 0 <= predicted_class_index < len(class_names):
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(np.max(score) * 100)
        return predicted_class_name, confidence
    else:
        return "Unknown", 0.0



# --- Navigation Menu Items ---
MENU_ITEMS = ["Home", "About", "Contact"]
DASHBOARD_ITEMS = [
    "Larval Diseases Data",
    "Pupae Defects Data",
    "Butterfly Life Cycle Data",
    "Butterfly Species Data",
    "Hostplants Info",
    "Tasks & Care Management",
]

# --- Functions to display content for each section ---
def display_home():
    st.title("🦋 Welcome to the Butterfly App")
    st.write("This app provides comprehensive information about butterflies, their care, diseases, life cycle, and features powerful **Image Classifiers** to identify various aspects of butterfly biology!")
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e0/Monarch_Butterfly_%28Danaus_plexippus%29_on_Lantana.jpg", use_container_width=True, caption="A Culture Butterfly in Marinduque")
    st.markdown("""
        <div style="padding: 20px; background-color: rgba(240, 242, 246, 0.7); border-radius: 10px; margin-top: 30px;">
            <h3>App Features:</h3>
            <ul>
                <li>Information on Larval Diseases and Pupae Defects</li>
                <li>Detailed view of Butterfly Life Cycle</li>
                <li>Data insights on various Butterfly and Moth Species</li>
                <li>Hostplants information for different species</li>
                <li>Task and Care activity management for butterfly farming</li>
                <li><b>New!</b> Upload an image to classify butterfly species, life stages, pupae defects, or larval diseases.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
 
def display_about():
    st.title("About Butterflies")
    st.write("Butterflies are fascinating insects with a complex life cycle and play a vital role in ecosystems as pollinators.")
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/6/68/Butterfly_beauty_01.jpg",
        caption="Beautiful Butterfly",
    )

    st.markdown("""
    <style>
        .card-banner {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-top: 30px;
            margin-bottom: 30px;
        }
        .col-card {
            flex: 1 1 calc(25% - 20px);
            min-width: 220px;
            max-width: 250px;
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            background-color: #fcfcfc;
            transition: transform 0.2s ease-in-out;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .col-card:hover {
            transform: translateY(-5px);
        }
        .banner {
            margin-bottom: 15px;
        }
        .img-banner {
            max-width: 80px;
            height: auto;
            border-radius: 50%;
            padding: 5px;
            background-color: #e0f7fa;
        }
        .title-banner {
            color: #333;
            font-size: 1.3em;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        .p-index {
            color: #0d0505;
            font-size: 0.9em;
            line-height: 1.5;
            flex-grow: 1;
        }
    </style>
    """, unsafe_allow_html=True)


    cards_data = [
        {
            "icon": "https://cdn-icons-png.flaticon.com/512/1057/1057398.png",
            "title": "Purpose",
            "text": "To address challenges in agriculture by continuously monitoring, measuring, and analyzing physical aspects and phenomena in complex, multivariate, and unpredictable ecosystems."
        },
        {
            "icon": "https://cdn-icons-png.flaticon.com/512/3759/3759048.png",
            "title": "Quality",
            "text": "As a farmer, the task is to culture butterflies with extra care management by maintaining indicator host plants for sustainability needs in butterfly farming or propagation."
        },
        {
            "icon": "https://cdn-icons-png.flaticon.com/512/917/917822.png",
            "title": "Function",
            "text": "To gain knowledge about Lepidoptera, cultured species are examined sequentially and adaptively identified. The system should precisely determine tasks and predict models for image segmentation, object detection, or classification."
        },
        {
            "icon": "https://cdn-icons-png.flaticon.com/512/3081/3081699.png",
            "title": "Elegant",
            "text": "Machine learning models, with their ability to adapt and learn from new data, can refine breeding strategies and improve quality performance, helping breeders stay competitive in dynamic biodiversity. Integrating ML in breeding is now essential."
        }
    ]

    cols = st.columns(len(cards_data))

    for i, card in enumerate(cards_data):
        with cols[i]:
            st.markdown(f"""
            <div class="col-card">
                <div class="banner">
                    <img src="{card['icon']}" class="img-banner" alt="{card['title']}">
                </div>
                <div class="card-app-banner">
                    <h2 class="title-banner">{card['title']}</h2>
                    <p class="p-index">{card['text']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_contact():
    st.title("Contact Us")
    st.write("If you have any questions, please contact us at contact@butterflyapp.com")
    st.markdown("""
        <div style="padding: 20px; background-color: rgba(240, 242, 246, 0.7); border-radius: 10px; margin-top: 30px;">
            <h3>Reach Out To Us!</h3>
            <p>We'd love to hear from you. Feel free to send us an email or connect on social media.</p>
            <ul>
                <li><b>Email:</b> <a href="mailto:insectconnectin@butterflyapp.com">insectconnection@butterflyapp.com</a></li>
                <li><b>Phone:</b> +63 (932) 881-1749</li>
                <li><b>Address:</b> 0184 Butterfly Lane, Gasan City, Marinduque 4905</li>
            </ul>
            <p>Follow us on:
                <a href="#" style="text-decoration: none; margin-left: 10px;">
                    <img src="https://img.icons8.com/fluent/48/000000/facebook-new.png" width="24" height="24" alt="Facebook">
                </a>
                <a href="#" style="text-decoration: none; margin-left: 5px;">
                    <img src="https://img.icons8.com/fluent/48/000000/twitter.png" width="24" height="24" alt="Twitter">
                </a>
                <a href="#" style="text-decoration: none; margin-left: 5px;">
                    <img src="https://img.icons8.com/fluent/48/000000/instagram-new.png" width="24" height="24" alt="Instagram">
                </a>
            </p>
        </div>
    """, unsafe_allow_html=True)

def display_larval_diseases_data():
    st.title("Larval Diseases Data")
    st.write("Learn about common diseases that affect butterfly larvae.")
    if not butterfly_larval_disease.empty:
        st.subheader("Disease Cases")
        st.dataframe(butterfly_larval_disease)

        st.subheader("Disease Distribution")
        fig, ax = plt.subplots()
        sns.barplot(x="disease_name", y="score", data=butterfly_larval_disease, palette="viridis", ax=ax)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No larval disease data available.")

def display_pupae_defects_data():
    st.title("Pupae Defects Data")
    st.write("Discover the different types of defects that can occur in pupae.")
    if not butterfly_pupae.empty:
        st.subheader("Defect Cases")
        st.dataframe(butterfly_pupae)

        st.subheader("Defect Distribution")
        fig, ax = plt.subplots()
        sns.barplot(x="defects_name", y="score", data=butterfly_pupae, palette="magma", ax=ax)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No pupae defects data available.")

def display_butterfly_data():
    st.title("Butterfly & Moth Species Data")
    st.write("Explore data related to various butterfly and moth species.")

    st.subheader("Species Information Table")
    if not butterfly_table.empty:
        st.dataframe(butterfly_table, use_container_width=True)
    else:
        st.info("No butterfly species table data available.")

    st.subheader("Species Image Counts (from butterfly_data.csv)")
    df_raw_butterfly_data = load_csv_data(CSV_FILE_BUTTERFLY_DATA)
    if not df_raw_butterfly_data.empty:
        with st.expander("Raw Data (from butterfly_data.csv)"):
            st.dataframe(df_raw_butterfly_data)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Species", y="Number of Images", data=df_raw_butterfly_data, palette="viridis", ax=ax)
        plt.xticks(rotation=75, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info(f"No species image count data found in {CSV_FILE_BUTTERFLY_DATA}.")

    st.subheader("Butterfly Species Detection Performance")
    if not butterfly_species_detected.empty:
        st.dataframe(butterfly_species_detected, use_container_width=True)
        # Assuming butterfly_species_detected has columns like 'Species', 'Correct Detected', 'Incorrect Detected'
        # To plot Correct vs Incorrect for each species
        fig, ax = plt.subplots(figsize=(10, 6))
        df_plot = butterfly_species_detected.melt(id_vars='Butterfly Species', value_vars=['Correct Detected', 'Incorrect Detected'],
                                                  var_name='Detection Type', value_name='Count')
        sns.barplot(x="Butterfly Species", y="Count", hue="Detection Type", data=df_plot, palette="Paired", ax=ax)
        plt.xticks(rotation=75, ha="right")
        plt.title("Correct vs. Incorrect Detections per Species")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No butterfly species detection performance data available.")


def display_butterfly_life_cycle_data():
    st.title("Butterfly Life Cycle Data")
    st.write("Explore the stages of a butterfly's life cycle.")

    st.subheader("Life Stages Overview")
    if not butterfly_stages.empty:
        with st.expander("Raw Data (from stages count.csv)"):
            st.dataframe(butterfly_stages)

        st.subheader("Butterfly Life Stages Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x="class_name", y="score", data=butterfly_stages, palette="viridis", ax=ax)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No butterfly life cycle data available.")


def display_hostplants_info():
    st.title("Hostplants Information")
    st.write("Select a butterfly species to view its host plants.")

    selected_species = st.selectbox(
        "Choose a Butterfly/Moth Species:",
        butterfly_species_names,
        key="hostplant_species_selector"
    )

    if selected_species:
        if selected_species in hostplants_info:
            st.markdown(f"**Hostplants for {selected_species}:**")
            st.info(hostplants_info[selected_species])
            st.markdown("""
            <div style="padding: 10px; background-color: rgba(230, 247, 255, 0.7); border-left: 5px solid #2196F3; margin-top: 20px;">
                <b>Note:</b> Host plants are crucial for butterfly egg-laying and larval development.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"Hostplant information not available for **{selected_species}**.")
    else:
        st.info("Please select a butterfly species from the dropdown to see its host plants.")


# Load existing tasks and care data from CSVs if they exist
@st.cache_data(show_spinner=False)
def load_data_from_csv_list(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            return pd.read_csv(file_path).to_dict(orient='records')
        except pd.errors.EmptyDataError:
            return []
    return []

# Initialize session state for tasks and care data
if "tasks" not in st.session_state:
    st.session_state.tasks = load_data_from_csv_list(CSV_FILE_TASKS)
if "care_data" not in st.session_state:
    st.session_state.care_data = load_data_from_csv_list(CSV_FILE_CARE)

def display_tasks_and_care():
    st.title("Tasks & Care Management")

    species_list = butterfly_species_names

    st.sidebar.subheader("Task and Care Management")
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

    if menu_selection == "Register Task":
        st.subheader("Register New Task")
        with st.form("task_registration_form", clear_on_submit=True):
            day = st.date_input("Select the Day", key="task_day")
            hour = st.time_input("Select the Hour", key="task_hour")
            activity = st.selectbox(
                "Choose Activity Type",
                [
                    "Harvesting Eggs", "Harvesting Pupae", "Harvesting Pupae and Eggs",
                    "Feeding Larvae", "Butterfly Foraging", "Cleaning Enclosures",
                    "Pest Control", "Health Check"
                ],
                key="task_activity"
            )
            details = st.text_area("Additional Details", key="task_details")
            species = st.selectbox("Select Species", species_list, key="task_species")

            submitted_task = st.form_submit_button("Add Task")

        if submitted_task:
            new_task = {
                "Day": str(day),
                "Hour": str(hour),
                "Activity": activity,
                "Details": details,
                "Species": species,
            }
            st.session_state.tasks.append(new_task)
            try:
                df = pd.DataFrame(st.session_state.tasks)
                df.to_csv(CSV_FILE_TASKS, index=False)
            # Ensure proper handling for empty CSV files (header issue)
            except Exception as e:
                st.error(f"Error saving task to CSV: {e}. Task added to session, but not saved persistently.")
                st.warning("If this is the first entry, manually ensure the CSV file is empty or has headers 'Day,Hour,Activity,Details,Species'.")
            st.success(f"Task added and saved: {new_task['Activity']} for {new_task['Species']} on {new_task['Day']}")

    elif menu_selection == "View Tasks":
        st.subheader("All Registered Tasks")
        if st.session_state.tasks:
            df_tasks = pd.DataFrame(st.session_state.tasks)
            st.dataframe(df_tasks, use_container_width=True)
        else:
            st.info("No tasks registered yet.")

    elif menu_selection == "View Task Distribution":
        st.subheader("Task Distribution Analysis")
        if st.session_state.tasks:
            df = pd.DataFrame(st.session_state.tasks)
            st.dataframe(df, use_container_width=True)

            st.markdown("---")
            st.subheader("Task Activity Distribution")
            activity_counts = df["Activity"].value_counts()
            fig_activity, ax_activity = plt.subplots(figsize=(10, 6))
            activity_counts.plot(kind='bar', ax=ax_activity, color='skyblue')
            ax_activity.set_title("Tasks Distribution by Activity")
            ax_activity.set_xlabel("Activity")
            ax_activity.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig_activity)

            st.markdown("---")
            st.subheader("Task Species Distribution")
            species_counts = df["Species"].value_counts()
            fig_species, ax_species = plt.subplots(figsize=(12, 7))
            species_counts.plot(kind='bar', ax=ax_species, color='lightgreen')
            ax_species.set_title("Tasks Distribution by Species")
            ax_species.set_xlabel("Species")
            ax_species.set_ylabel("Count")
            plt.xticks(rotation=75, ha="right")
            plt.tight_layout()
            st.pyplot(fig_species)

        else:
            st.info("No tasks registered yet to display distribution.")

    elif menu_selection == "Record Care Activity":
        st.subheader("Record New Care Activity")
        with st.form("care_activity_form", clear_on_submit=True):
            care_day = st.date_input("Select Care Day", key="care_day")
            care_hour = st.time_input("Select Care Hour", key="care_hour")
            care_species = st.selectbox("Select Species", species_list, key="care_species")
            care_activity = st.text_area("Care Activity Description", height=100, key="care_activity")

            submitted_care = st.form_submit_button("Record Care")

        if submitted_care:
            new_care_data = {
                "Day": str(care_day),
                "Hour": str(care_hour),
                "Species": care_species,
                "Activity": care_activity,
            }
            st.session_state.care_data.append(new_care_data)
            try:
                df_care = pd.DataFrame(st.session_state.care_data)
                df_care.to_csv(CSV_FILE_CARE, index=False)
            # Ensure proper handling for empty CSV files (header issue)
            except Exception as e:
                st.error(f"Error saving care data to CSV: {e}. Care activity added to session, but not saved persistently.")
                st.warning("If this is the first entry, manually ensure the CSV file is empty or has headers 'Day,Hour,Species,Activity'.")
            st.success(f"Care activity recorded and saved: {new_care_data['Activity']} for {new_care_data['Species']} on {new_care_data['Day']}")

    elif menu_selection == "View Care Activities":
        st.subheader("All Recorded Care Activities")
        if st.session_state.care_data:
            df_care = pd.DataFrame(st.session_state.care_data)
            st.dataframe(df_care, use_container_width=True)
        else:
            st.info("No care activities recorded yet.")

# --- Main App Logic ---
st.sidebar.title("Menu")
selected_menu_item = st.sidebar.radio("Go to", MENU_ITEMS)

if selected_menu_item == "Home":
    display_home()
elif selected_menu_item == "About":
    display_about()
elif selected_menu_item == "Contact":
    display_contact()

st.sidebar.title("Dashboard Menu")
selected_dashboard_item = st.sidebar.radio("Explore Data & Tools", DASHBOARD_ITEMS)

if selected_dashboard_item == "Larval Diseases Data":
    display_larval_diseases_data()
elif selected_dashboard_item == "Pupae Defects Data":
    display_pupae_defects_data()
elif selected_dashboard_item == "Butterfly Life Cycle Data":
    display_butterfly_life_cycle_data()
elif selected_dashboard_item == "Butterfly Species Data":
    display_butterfly_data()
elif selected_dashboard_item == "Hostplants Info":
    display_hostplants_info()
elif selected_dashboard_item == "Tasks & Care Management":
    display_tasks_and_care()