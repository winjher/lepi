import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Set page configuration at the very beginning
# st.set_page_config(layout="wide", page_title="Butterfly App")
#st.image('icon/bgbutterfly.jpg', width=200)
# --- Function to set background image ---
def set_background_image(image_url):
    """
    Sets a background image for the Streamlit application.

    Args:
        image_url (str): The URL of the image to use as a background.
                         This can be a local file path (e.g., "path/to/your/image.jpg")
                         or a URL from the web.
    """
    st.markdown('<div class="centered-image">', unsafe_allow_html=True)
    st.image('icon/bgbutterfly.jpg', width=800) # st.image still controls initial width
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover; /* Covers the entire area */
            background-repeat: no-repeat; /* Prevents image repetition */
            background-attachment: fixed; /* Keeps image fixed when scrolling */
            background-position: center; /* Centers the background image */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Apply the background image ---
# Ensure 'icon/bgbutterfly.jpg' is in the correct path relative to your script
set_background_image('..icon/bgbutterfly.jpg')


# --- Define Paths ---
MODEL_DIR = './model/' # Directory where your models are saved
IMAGE_DIR = './butterfly_photos/butterfly/' # Base directory for your images (for displaying species names)
DATA_DIR = './Data/' # Directory for your CSV data files
UPLOAD_DIR = './upload/' # Directory for uploaded images
butterfly_count = pd.read_csv("Data/butterfly_count.csv")
butterfly_table = pd.read_csv("Data/butterfly_table_species.csv")
butterfly_stages = pd.read_csv("Data/stages count.csv")
butterfly_pupae = pd.read_csv("Data/pupae_defects_records.csv")
butterfly_larval_disease = pd.read_csv("Data/larval_disease_records.csv")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True) # Ensure upload directory exists

# Define CSV file names for task management and care activities
CSV_FILE_TASKS = os.path.join(DATA_DIR, "tasks.csv")
CSV_FILE_CARE = os.path.join(DATA_DIR, "care_data.csv")
CSV_FILE_BUTTERFLY_DATA = os.path.join(DATA_DIR, "butterfly_data.csv") # Define path for butterfly_data.csv
CSV_FILE_STAGES_DATA = os.path.join(DATA_DIR, "Stages.csv") # Define path for Stages.csv
CSV_FILE_BUTTERFLY_COUNT = butterfly_count # Define path for Stages count.csv
CSV_FILE_BUTTERFLY_TABLE = butterfly_table
# --- Dummy data for CSVs if they don't exist ---
def create_dummy_csv(file_path, df_columns, data_rows):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        dummy_df = pd.DataFrame(data_rows, columns=df_columns)
        dummy_df.to_csv(file_path, index=False)
        st.info(f"Created dummy file: {file_path}")

# Dummy data for butterfly_data.csv
dummy_butterfly_species_data = [
    {"Species": "Butterfly-Clippers", "Number of Images": 10},
    {"Species": "Butterfly-Common Jay", "Number of Images": 15},
    {"Species": "Butterfly-Common Lime", "Number of Images": 12},
    {"Species": "Butterfly-Common Mime", "Number of Images": 18},
    {"Species": "Butterfly-Common Mormon", "Number of Images": 20},
    {"Species": "Butterfly-Emerald Swallowtail", "Number of Images": 8},
    {"Species": "Butterfly-Golden Birdwing", "Number of Images": 7},
    {"Species": "Butterfly-Gray Glassy Tiger", "Number of Images": 11},
    {"Species": "Butterfly-Great Eggfly", "Number of Images": 14},
    {"Species": "Butterfly-Great Yellow Mormon", "Number of Images": 9},
    {"Species": "Butterfly-Paper Kite", "Number of Images": 16},
    {"Species": "Butterfly-Pink Rose", "Number of Images": 6},
    {"Species": "Butterfly-Plain Tiger", "Number of Images": 13},
    {"Species": "Butterfly-Red Lacewing", "Number of Images": 5},
    {"Species": "Butterfly-Scarlet Mormon", "Number of Images": 10},
    {"Species": "Butterfly-Tailed Jay", "Number of Images": 12},
    {"Species": "Moth-Atlas", "Number of Images": 4},
    {"Species": "Moth-Giant Silk", "Number of Images": 3},
]

create_dummy_csv(CSV_FILE_BUTTERFLY_DATA, ["Species", "Number of Images"], dummy_butterfly_species_data)




# Dummy data for Stages.csv
create_dummy_csv(CSV_FILE_STAGES_DATA,
                  ["Stages Type", "Number of Stages"],
                  [{"Stages Type": "Eggs", "Number of Stages": 100},
                   {"Stages Type": "Larvae", "Number of Stages": 150},
                   {"Stages Type": "Pupae", "Number of Stages": 80},
                   {"Stages Type": "Adult Butterfly", "Number of Stages": 50}])

create_dummy_csv(CSV_FILE_TASKS,
                 ["Day", "Hour", "Activity", "Details", "Species"],
                 []) # Start with an empty tasks CSV

create_dummy_csv(CSV_FILE_CARE,
                 ["Day", "Hour", "Species", "Activity"],
                 []) # Start with an empty care_data CSV

# --- Load Models (once) ---
# Each model needs its own @st.cache_resource decorated function and variable
@st.cache_resource
def load_butterfly_species_model():
    model_path = os.path.join(MODEL_DIR, 'model_Butterfly_Species.h5')
    if not os.path.exists(model_path):
        st.error(f"Butterfly Species Model not found at: {model_path}. Please ensure the model is saved there.")
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading Butterfly Species Model: {e}")
        return None

@st.cache_resource
def load_lifestages_model():
    model_path = os.path.join(MODEL_DIR, 'model_Life_Stages.h5')
    if not os.path.exists(model_path):
        st.error(f"Life Stages Model not found at: {model_path}. Please ensure the model is saved there.")
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading Life Stages Model: {e}")
        return None

@st.cache_resource
def load_pupaedefects_model():
    model_path = os.path.join(MODEL_DIR, 'model_Pupae_Defects.h5')
    if not os.path.exists(model_path):
        st.error(f"Pupae Defects Model not found at: {model_path}. Please ensure the model is saved there.")
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading Pupae Defects Model: {e}")
        return None

@st.cache_resource
def load_larvaldiseases_model():
    model_path = os.path.join(MODEL_DIR, 'model_Larval_Diseases.h5')
    if not os.path.exists(model_path):
        st.error(f"Larval Diseases Model not found at: {model_path}. Please ensure the model is saved there.")
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading Larval Diseases Model: {e}")
        return None

# Load all models at the start
butterfly_species_model = load_butterfly_species_model()
lifestages_model = load_lifestages_model()
pupaedefects_model = load_pupaedefects_model()
larvaldiseases_model = load_larvaldiseases_model()


# --- Define Class Names for Each Model ---
# It's crucial that these match the order your models were trained on!
# Infer from directory for butterfly_species, fallback to default if not found
try:
    _species_names = sorted([d for d in os.listdir(os.path.join(IMAGE_DIR, "butterfly_photos")) if os.path.isdir(os.path.join(IMAGE_DIR, "butterfly_photos", d))])
    if not _species_names:
        raise FileNotFoundError # Trigger fallback
    butterfly_species_names = _species_names
except FileNotFoundError:
    #st.info("Using default butterfly species names. Ensure your `butterfly_photos/butterfly` directory is correctly structured for dynamic loading.")
    butterfly_species_names = [
        "Butterfly-Clippers", "Butterfly-Common Jay", "Butterfly-Common Lime",
        "Butterfly-Common Mime", "Butterfly-Common Mormon", "Butterfly-Emerald Swallowtail",
        "BUtterfly-Golden Birdwing", "Butterfly-Gray Glassy Tiger", "Butterfly-Great Eggfly",
        "Butterfly-Great Yellow Mormon", "Butterfly-Paper Kite", "Butterfly-Pink Rose",
        "Butterfly-Plain Tiger", "Butterfly-Red Lacewing", "Butterfly-Scarlet Mormon",
        "Butterfly-Tailed Jay", "Moth-Atlas", "Moth-Giant Silk",
    ]

lifestages_names = ["Eggs", "Larvae", "Pupae", "Adult Butterfly"]
pupaedefects_names = ["Ant bites", "Deformed body", "Healthy Pupae", "Old Pupa", "Overbend", "Stretch abdomen"]
larvaldiseases_names = ["Anaphylaxis Infection", "Gnathostomiasis", "Nucleopolyhedrosis"]

# --- Hostplants Information (Unified) ---
# This dictionary now uses the `butterfly_species_names` as keys for consistency
# Make sure the order or indexing aligns with your species classifier
hostplants_info = {
    "Butterfly-Clippers": "Wild Cucumber.",
    "Butterfly-Common Jay": "'Avocado Tree', 'Soursop', 'Sugar Apple', 'Amuyon', 'Indian Tree'.",
    "Butterfly-Common Lime": "'Limeberry', 'Calamondin', 'Pomelo', 'Sweet Orange', 'Calamansi'.",
    "Butterfly-Common Mime": "'Clover Cinnamon', 'Wild Cinnamon'.",
    "Butterfly-Common Mormon": "'Limeberry', 'Calamondin', 'Pomelo', 'Sweet Orange', 'Calamansi', 'Lemoncito'.",
    "Butterfly-Emerald Swallowtail": "'Curry Leafs', 'Pink Lime-Berry Tree'.",
    "BUtterfly-Golden Birdwing": "'Dutchman Pipe', 'Indian Birthwort'.", # Corrected typo in Golden Birdwing
    "Butterfly-Gray Glassy Tiger": "'Limeberry', 'Calamondin', 'Pomelo', 'Sweet Orange', 'Calamansi'.", # This seems like a duplicate for Lime, please verify
    "Butterfly-Great Eggfly": "Dutchman Pipe', 'Indian Birthwort'.",
    "Butterfly-Great Yellow Mormon": "'Sweet Potato', 'Water Spinach'.",
    "Butterfly-Paper Kite": "'Common Skillpod'.",
    "Butterfly-Pink Rose": "'Dutchman Pipe', 'Indian Birthwort'.",
    "Butterfly-Plain Tiger": "'Crown flower', 'Giant Milkweed'.",
    "Butterfly-Red Lacewing": "'Wild Bush Passion Fruits'.",
    "Butterfly-Scarlet Mormon": "'Calamondin', 'Pomelo', 'Sweet Orange', 'Calamansi'.",
    "Butterfly-Tailed Jay": "'Avocado Tree', 'Soursop', 'Sugar Apple', 'Amuyon', 'Indian Tree'.",
    "Moth-Atlas": "'Gmelina Tree', 'Soursop'.",
    "Moth-Giant Silk": "'Curry Leafs'."
}

# --- Load image counts for species from CSV ---
@st.cache_data(show_spinner=False)
def load_species_image_counts(file_path):
    try:
        df = pd.read_csv(file_path)
        # Convert to dictionary for easy lookup: {species_name: count}
        return df.set_index('Species')['Number of Images'].to_dict()
    except FileNotFoundError:
        st.error(f"Species image count data not found at: {file_path}. Using dummy counts.")
        # Fallback to dummy data if CSV not found or empty
        return {item["Species"]: item["Number of Images"] for item in dummy_butterfly_species_data}
    except pd.errors.EmptyDataError:
        st.warning(f"Species image count CSV at {file_path} is empty. Using dummy counts.")
        return {item["Species"]: item["Number of Images"] for item in dummy_butterfly_species_data}
    except Exception as e:
        st.error(f"Error loading species image counts: {e}. Using dummy counts.")
        return {item["Species"]: item["Number of Images"] for item in dummy_butterfly_species_data}

species_image_counts = load_species_image_counts(CSV_FILE_BUTTERFLY_DATA)


# --- Generic Prediction Function for any Classifier ---
def classify_image(image_file, model, class_names, img_size=(180, 180)):
    if model is None:
        return "Model Not Loaded", 0.0

    img = Image.open(image_file).resize(img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch dimension
    img_array = img_array / 255.0 # Normalize image data if your model expects it

    predictions = model.predict(img_array)
    # Check if the model output is logits (requires softmax) or probabilities (already softmaxed)
    # If your model's last layer is softmax, skip tf.nn.softmax here.
    # Given the previous problem with sigmoid and CategoricalCrossentropy,
    # it's safer to apply softmax if the model output is not already probabilities sum to 1.
    # A robust way is to check the model's output activation, but without that info,
    # applying softmax to logits is generally correct.
    if predictions.shape[-1] > 1: # Multi-class
        score = tf.nn.softmax(predictions[0])
    else: # Binary or a single output (e.g., sigmoid), assume it's already a probability
        score = predictions[0]

    predicted_class_index = np.argmax(score)

    if 0 <= predicted_class_index < len(class_names):
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(np.max(score) * 100) # Ensure confidence is a standard float
        return predicted_class_name, confidence
    else:
        return "Unknown", 0.0


# --- Navigation Menu Items ---
MENU_ITEMS = [
    "Home",
    "About",
    "Contact",
    "Larval Diseases Data", # Clarified
    "Pupae Defects Data", # Clarified
    "Butterfly Life Cycle Data", # Clarified
    "Butterfly Species Data",
    "Hostplants Info", # NEW MENU ITEM
    "Tasks & Care Management",
    "Image Classifiers" # Unified menu for all classifiers
]

# --- Functions to display content for each section ---
def display_home():
    st.title("🦋 Welcome to the Butterfly App")
    st.write("This app provides comprehensive information about butterflies, their care, diseases, life cycle, and features powerful **Image Classifiers** to identify various aspects of butterfly biology!")
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e0/Monarch_Butterfly_%28Danaus_plexippus%29_on_Lantana.jpg", use_container_width=True, caption="A Culture Butterfly in Marinduque")
    st.markdown("""
        <div style="padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-top: 30px;">
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
            color: #555;
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
        <div style="padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-top: 30px;">
            <h3>Reach Out To Us!</h3>
            <p>We'd love to hear from you. Feel free to send us an email or connect on social media.</p>
            <ul>
                <li><b>Email:</b> <a href="mailto:info@butterflyapp.com">info@butterflyapp.com</a></li>
                <li><b>Phone:</b> +1 (123) 456-7890</li>
                <li><b>Address:</b> 123 Butterfly Lane, Entomology City, BC 45678</li>
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
    # larval_disease_names_data = ["Anaphylaxis Infection", "Gnathostomiasis", "Healthy","Nucleopolyhedrosis"]
    # number_of_cases = [120, 220, 500, 50]
    
    st.subheader("Disease Cases")
    #df_diseases = pd.DataFrame({"Disease Name": larval_disease_names_data, "Number of Cases": number_of_cases})
    st.dataframe(butterfly_larval_disease)

    st.subheader("Disease Distribution")
    fig, ax = plt.subplots()
    sns.barplot(x="disease_name", y="score", data=butterfly_larval_disease, palette="viridis", ax=ax)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

def display_pupae_defects_data():
    st.title("Pupae Defects Data")
    st.write("Discover the different types of defects that can occur in pupae.")
    # pupae_defects_names_data = ["Ant bites", "Deformed body", "Healthy Pupae", "Old Pupa", "Overbend", "Stretch abdomen"]
    # number_of_defects = [50, 30, 200, 25, 15, 40]

    # st.subheader("Defect Cases")
    # df_defects = pd.DataFrame(
    #     {"Defect Type": pupae_defects_names_data, "Number of Defects/Healthy": number_of_defects}
    # )
    df_defects= butterfly_pupae
    st.dataframe(df_defects)

    st.subheader("Defect Distribution")
    fig, ax = plt.subplots()
    sns.barplot(x="defects_name", y="score", data=df_defects, palette="magma", ax=ax)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

def display_butterfly_data():
    st.title("Butterfly & Moth Species Data")
    st.write("Explore data related to various butterfly and moth species.")

    # Using the names loaded for the classifier to maintain consistency
    #species_of_butterfly = butterfly_species_names
    # Placeholder for actual data or load from CSV if available
    # Using the loaded species_image_counts for consistency
    #total_number_of_images = [species_image_counts.get(s, 0) for s in species_of_butterfly]

  
    st.subheader("Species Information")
    df_butterfly_display = butterfly_count
    df_butterfly_display_table = butterfly_table
    #pd.DataFrame({"Species": species_of_butterfly, "Number of Images": total_number_of_images})
    with st.expander("Raw Data (from butterfly_data.csv)"):
        try:
            df_raw_butterfly = pd.read_csv(CSV_FILE_BUTTERFLY_DATA)
            st.dataframe(df_raw_butterfly)
        except FileNotFoundError:
            st.error(f"{CSV_FILE_BUTTERFLY_DATA} not found. Please ensure it exists.")
        except pd.errors.EmptyDataError:
            st.warning(f"{CSV_FILE_BUTTERFLY_DATA} is empty.")
            st.dataframe(pd.DataFrame())

    st.subheader("Species Distribution (Display Data)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Species", y="Number of Images", data=df_raw_butterfly, palette="viridis", ax=ax)
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(df_butterfly_display)

    st.subheader("Species Score (Display Data Table)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="species", y="score", data=df_butterfly_display_table, palette="viridis", ax=ax)
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    
    st.dataframe(df_butterfly_display_table)

def display_butterfly_life_cycle_data():
    st.title("Butterfly Life Cycle Data")
    st.write("Explore the stages of a butterfly's life cycle.")
    life_cycle_data = ["Eggs", "Larvae", "Pupae", "Adult Butterfly"]
    # Load from Stages.csv
    try:
        df_stages_raw = pd.read_csv(CSV_FILE_STAGES_DATA) # Using CSV_FILE_STAGES_DATA
        # Ensure order matches life_cycle_data for consistent plotting
        number_of_stages = [df_stages_raw[df_stages_raw['Stages Type'] == stage]['Number of Stages'].sum() for stage in life_cycle_data]
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.warning(f"Could not load {CSV_FILE_STAGES_DATA}. Using dummy data for display.") # Corrected reference
        number_of_stages = [230, 340, 250, 500] # Fallback dummy data, uncommented and active


    st.subheader("Life Stages Overview")
    #df_stages_display = pd.DataFrame({"Stages Type": life_cycle_data, "Number of Stages": number_of_stages})
    butterfly_stages
    with st.expander("Raw Data (from Stages.csv)"):
        try:
            df_raw_stages = pd.read_csv(CSV_FILE_STAGES_DATA)
            st.dataframe(df_raw_stages)
        except FileNotFoundError:
            st.error(f"{CSV_FILE_STAGES_DATA} not found. Please ensure it exists.")
        except pd.errors.EmptyDataError:
            st.warning(f"{CSV_FILE_STAGES_DATA} is empty.")
            st.dataframe(pd.DataFrame())

    st.subheader("Butterfly Life Cycle Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Stage", y="Count", data=butterfly_stages, palette="viridis", ax=ax)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(butterfly_stages)

# New function to display hostplants information
def display_hostplants_info():
    st.title("Hostplants Information")
    st.write("Select a butterfly species to view its host plants.")

    # Get the list of butterfly species from the `butterfly_species_names`
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
            <div style="padding: 10px; background-color: #e6f7ff; border-left: 5px solid #2196F3; margin-top: 20px;">
                <b>Note:</b> Host plants are crucial for butterfly egg-laying and larval development.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"Hostplant information not available for **{selected_species}**.")
    else:
        st.info("Please select a butterfly species from the dropdown to see its host plants.")


# Load existing tasks and care data from CSVs if they exist
@st.cache_data(show_spinner=False)
def load_data_from_csv(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            return pd.read_csv(file_path).to_dict(orient='records')
        except pd.errors.EmptyDataError:
            return []
    return []

# Initialize session state for tasks and care data
if "tasks" not in st.session_state:
    st.session_state.tasks = load_data_from_csv(CSV_FILE_TASKS)
if "care_data" not in st.session_state:
    st.session_state.care_data = load_data_from_csv(CSV_FILE_CARE)

def display_tasks_and_care():
    st.title("Tasks & Care Management")

    species_list = butterfly_species_names # Use the dynamically loaded or default species names

    st.sidebar.subheader("Task Management Sub-menu")
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
                    "Harvesting Eggs",
                    "Harvesting Pupae",
                    "Harvesting Pupae and Eggs",
                    "Feeding Larvae",
                    "Butterfly Foraging",
                    "Cleaning Enclosures",
                    "Pest Control",
                    "Health Check"
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
                st.success(f"Task added and saved: {new_task['Activity']} for {new_task['Species']} on {new_task['Day']}")
            except Exception as e:
                st.error(f"Error saving task to CSV: {e}. Task added to session, but not saved persistently.")

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
                st.success(f"Care activity recorded and saved: {new_care_data['Activity']} for {new_care_data['Species']} on {new_care_data['Day']}")
            except Exception as e:
                st.error(f"Error saving care data to CSV: {e}. Care activity added to session, but not saved persistently.")

    elif menu_selection == "View Care Activities":
        st.subheader("All Recorded Care Activities")
        if st.session_state.care_data:
            df_care = pd.DataFrame(st.session_state.care_data)
            st.dataframe(df_care, use_container_width=True)
        else:
            st.info("No care activities recorded yet.")

def display_image_classifiers():
    st.title("🔬 Image Classifiers")
    st.write("Select a classifier to analyze your butterfly images.")

    classifier_choice = st.sidebar.radio(
        "Choose Classifier Type",
        ["Butterfly Species", "Life Stages", "Pupae Defects", "Larval Diseases"]
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(f"Upload an image for {classifier_choice} classification...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        st.write("")
        st.write("Classifying...")

        model_to_use = None
        class_names_to_use = []

        if classifier_choice == "Butterfly Species":
            model_to_use = butterfly_species_model
            class_names_to_use = butterfly_species_names
        elif classifier_choice == "Life Stages":
            model_to_use = lifestages_model
            class_names_to_use = lifestages_names
        elif classifier_choice == "Pupae Defects":
            model_to_use = pupaedefects_model
            class_names_to_use = pupaedefects_names
        elif classifier_choice == "Larval Diseases":
            model_to_use = larvaldiseases_model
            class_names_to_use = larvaldiseases_names

        if model_to_use:
            try:
                predicted_class, confidence = classify_image(uploaded_file, model_to_use, class_names_to_use)
                st.success(f"**{classifier_choice} Prediction:** **{predicted_class}** (Confidence: {confidence:.2f}%)")

            except Exception as e:
                st.error(f"Error during classification: {e}")
        else:
            st.warning(f"The {classifier_choice} model is not loaded. Please check the model path and ensure the model file exists.")

# --- Main App Logic ---
st.sidebar.title("Navigation")
selected_menu_item = st.sidebar.radio("Go to", MENU_ITEMS)

if selected_menu_item == "Home":
    display_home()
elif selected_menu_item == "About":
    display_about()
elif selected_menu_item == "Contact":
    display_contact()
elif selected_menu_item == "Larval Diseases Data":
    display_larval_diseases_data()
elif selected_menu_item == "Pupae Defects Data":
    display_pupae_defects_data()
elif selected_menu_item == "Butterfly Life Cycle Data":
    display_butterfly_life_cycle_data()
elif selected_menu_item == "Butterfly Species Data":
    display_butterfly_data()
elif selected_menu_item == "Hostplants Info":
    display_hostplants_info()
elif selected_menu_item == "Tasks & Care Management":
    display_tasks_and_care()
elif selected_menu_item == "Image Classifiers":
    display_image_classifiers()

# Embed custom CSS
st.markdown(
    """
    <style>
    .centered-image {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .centered-image img {
        display: block; /* Remove extra space below image */
        max-width: 800px; /* Set max width if needed */
        width: 100%; /* Make it responsive up to max-width */
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the image within a div that uses the custom CSS class
