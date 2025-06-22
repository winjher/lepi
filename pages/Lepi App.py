import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd # Import pandas for data logging
import datetime # Import datetime for timestamps


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
try:
    set_background_image('icon/bbg.png')
except FileNotFoundError:
    st.warning("Background image 'icon/bg.jpg' not found. Please ensure it's in the correct path.")

# --- Glasmorphism CSS ---
# Define general styles for glasmorphism elements within the app
st.markdown(
    """
    <style>
    /* General body/text styling for better readability on blurred backgrounds */
    body {
        color: #333333; /* Darker text for better contrast */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #222222;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1); /* Subtle text shadow */
    }
    .stTextInput label, .stSelectbox label, .stRadio label, .stFileUploader label, .stCameraInput label {
        color: #333333;
        font-weight: bold;
    }
    .stButton>button {
        background-color: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #333333;
        transition: all 0.3s ease;
        border-radius: 8px; /* Rounded buttons */
    }
    .stButton>button:hover {
        background-color: rgba(255, 255, 255, 0.3);
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
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
    /* Apply glasmorphism to the main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.15); /* Slightly less opaque for the whole main area */
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 20px; /* Space from the bottom if page is short */
    }

    /* Apply glasmorphism to the sidebar */
    /* This class might change with Streamlit updates, inspect with browser dev tools */
    .css-pkaj6s { /* Common class for the sidebar parent div */
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 20px;
        margin-right: 10px; /* Adjust spacing from main content if needed */
    }
    
    /* Specific glasmorphism container for individual elements if needed (e.g., st.image, st.success) */
    .glasmorphism-card {
        background: rgba(255, 255, 255, 0.25); /* Slightly more opaque for inner cards */
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 15px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Configuration and Model Loading ---

# Define the directory where your models are stored
MODEL_DIR = './model' # Assuming models are in a 'model' subdirectory
IMAGE_SIZE = (180, 180) # Model's expected input size

# Ensure the model directory exists
if not os.path.exists(MODEL_DIR):
    st.error(f"Model directory not found: {MODEL_DIR}. Please create it and place your models inside.")
    st.stop()

# Ensure the data directory exists for logging
DATA_DIR = './Data'
os.makedirs(DATA_DIR, exist_ok=True)


# --- Load Models (once) ---
# Each model needs its own @st.cache_resource decorated function and variable
@st.cache_resource
def load_model(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}. Please ensure the model is saved there.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading {model_name} Model: {e}")
        return None

# Load all models at the start
butterfly_species_model = load_model('model_Butterfly_Species.h5')
lifestages_model = load_model('model_Life_Stages.h5')
pupaedefects_model = load_model('model_Pupae_Defects.h5')
larvaldiseases_model = load_model('model_Larval_Diseases.h5')


# --- Define Class Names and Associated Information for Each Model ---
# It's crucial that these match the order your models were trained on!
# You will need to populate the detailed info for each class.

butterfly_species_info = {
    "Butterfly-Clippers": {"scientific_name": "Parthenos sylvia", "family": "Nymphalidae", "discovered":"Carl Peter Thunberg, Cramer","year":"1776", "description":"Forewing triangular; costa very slightly curved, apex rounded, exterior margin oblique and slightly scalloped, posterior margin short, angle convex; "},
    "Butterfly-Common Jay": {"scientific_name": "Graphium doson", "family": "Papilionidae", "discovered":"C. & R. Felder","year":"1864", "description":""},
    "Butterfly-Common Lime": {"scientific_name": "Papilio demoleus", "family": "Papilionidae", "discovered":"Linnaeus","year":"1758", "description":"The butterfly is tailless and has a wingspan 80–100 mm,the butterfly has a large number of irregular spots on the wing."},
    "Butterfly-Common Mime": {"scientific_name": "Papilio clytia", "family": "Papilionidae", "discovered":"Linnaeus","year":"1758", "description":" It's a black-bodied swallowtail and a good example of Batesian mimicry, meaning it mimics the appearance of other distasteful butterflies. "},
    "Butterfly-Common Mormon": {"scientific_name": "Papilio polytes", "family": "Papilionidae","discovered":"Linnaeus","year":"1758", "description":" "},
    "Butterfly-Emerald Swallowtail": {"scientific_name": "Papilio palinurus", "family": "Papilionidae", "discovered":"Fabricius","year":"1787", "description":""},
    "Butterfly-Golden Birdwing": {"scientific_name": "Troides rhadamantus", "family": "Papilionidae", "discovered":"H. Lucas","year":"1835", "description":""},
    "Butterfly-Gray Glassy Tiger": {"scientific_name": "Ideopsis juventa", "family": "Nymphalidae", "discovered":"Cramer","year":"1777", "description":""},
    "Butterfly-Great Eggfly": {"scientific_name": "Hypolimnas bolina", "family": "Nymphalidae", "discovered":"Linnaeus","year":"1758", "description":""},
    "Butterfly-Great Yellow Mormon": {"scientific_name": "Papilio lowi", "family": "Papilionidae", "discovered":"","year":"", "description":""},
    "Butterfly-Paper Kite": {"scientific_name": "Idea leuconoe", "family": "Nymphalidae", "discovered":"Rothschild","year":"1895", "description":""},
    "Butterfly-Pink Rose": {"scientific_name": "Pachliopta kotzebuea", "family": "Papilionidae", "discovered":"Escholtz","year":"1821", "description":""},
    "Butterfly-Plain Tiger": {"scientific_name": "Danaus chrysippus", "family": "Nymphalidae", "discovered":"Hulstaert","year":"1931", "description":""},
    "Butterfly-Red Lacewing": {"scientific_name": "Cethosia biblis", "family": "Nymphalidae", "discovered":"Drury","year":"1773", "description":""},
    "Butterfly-Scarlet Mormon": {"scientific_name": "Papilio rumanzovia", "family": "Papilionidae", "discovered":"Eschscholtz","year":"1821", "description":""},
    "Butterfly-Tailed Jay": {"scientific_name": "Graphium agamemnon", "family": "Papilionidae", "discovered":"Linnaeus","year":"1758", "description":""},
    "Moth-Atlas": {"scientific_name": "Attacus atlas", "family": "Saturniidae","discovered":"Linnaeus","year":"1758", "description":""},
    "Moth-Giant Silk": {"scientific_name": "Samia cynthia", "family": "Saturniidae", "discovered":"Hubner","year":"1819", "description":""},
}
butterfly_species_names = list(butterfly_species_info.keys()) # Ensure order if your model was trained on sorted keys

lifestages_info = {
    "Butterfly": {"stages_info": "Reproductive stage, winged insect capable of flight."},
    "Eggs": {"stages_info": "Early developmental stage, typically laid on host plants."},
    "Larvae": {"stages_info": "Caterpillar stage, primary feeding and growth phase."},
    "Pupae": {"stages_info": "Chrysalis (butterfly) or cocoon (moth) stage, metamorphosis occurs."},
    
}
lifestages_names = list(lifestages_info.keys())

pupaedefects_info = {
    "Ant bites": {"quality_info": "Indicates ant damage, can lead to pupae death or malformation."},
    "Deformed body": {"quality_info": "Physical deformities, may indicate poor health or environmental stress."},
    "Healthy Pupae": {"quality_info": "No visible defects, good potential for adult emergence."},
    "Old Pupa": {"quality_info": "Pupae nearing emergence or past its prime, may be discolored or shriveled."},
    "Overbend": {"quality_info": "Abnormal curvature, can impede proper development."},
    "Stretch abdomen": {"quality_info": "Abdomen appears stretched or elongated, potentially due to stress or disease."},
}
pupaedefects_names = list(pupaedefects_info.keys())

larvaldiseases_info = {
    "Anaphylaxis Infection": {"treatment_info": "Seek entomologist advice; isolate infected larvae. No specific treatment for severe cases."},
    "Gnathostomiasis": {"treatment_info": "Parasitic infection. Isolate, remove parasites if visible, improve hygiene."},
    "Healthy": {"treatment_info": "Larva appears healthy with no signs of disease."},
    "Nucleopolyhedrosis": {"treatment_info": "Highly contagious viral disease. Isolate and destroy infected larvae to prevent spread. Disinfect rearing areas."},
}
larvaldiseases_names = list(larvaldiseases_info.keys())


# --- Classification Function ---
def classify_image(image_file, model, class_names, details_dict=None):
    """
    Classifies an image using the specified Keras model and class names.
    Optionally returns additional details based on the classified class.

    Args:
        image_file: A file-like object from st.file_uploader or bytes from st.camera_input.
        model: The loaded TensorFlow/Keras model.
        class_names: A list of class names corresponding to the model's output.
        details_dict: A dictionary mapping class names to additional information.

    Returns:
        A dictionary containing class name, score, and any additional details, or None on error.
    """
    if model is None:
        return None

    try:
        image = Image.open(image_file)
        img_resized = image.resize(IMAGE_SIZE)
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0) # Add batch dimension

        predictions = model.predict(img_array)
        result = tf.nn.softmax(predictions[0])
        predicted_class_index = np.argmax(result)
        predicted_score = np.max(result).item() * 100

        if predicted_class_index < len(class_names):
            predicted_class_name = class_names[predicted_class_index]
            result_data = {
                "class_name": predicted_class_name,
                "score": predicted_score,
                "index": predicted_class_index
            }
            if details_dict and predicted_class_name in details_dict:
                result_data.update(details_dict[predicted_class_name])
            return result_data
        else:
            return {"class_name": "Unknown Class (Index out of bounds)", "score": 0.0, "index": predicted_class_index}
    except Exception as e:
        st.error(f"Error during image classification: {e}")
        return None

# --- Streamlit UI Layout ---
st.title("🦋 Lepi App")
st.write("Upload an image or capture from your webcam to classify butterflies, their life stages, or identify pupae defects and larval diseases.")

# Sidebar for navigation
st.sidebar.title("Classify")
selected_menu_item = st.sidebar.radio(
    "Go to Menu",
    ["Image Classifiers"] # Simplified for this specific request
)

# --- Main Content Area based on Mode Selection ---
if selected_menu_item == "Image Classifiers":
    st.header("🔬 Image Classifiers")
    st.write("Select a classifier to analyze your butterfly images.")

    classifier_choice = st.radio(
        "Choose Classifier Type",
        ["Butterfly Species", "Life Stages", "Pupae Defects", "Larval Diseases"]
    )

    st.markdown("---")

    # Input method selection for classifiers
    input_method = st.radio("Choose an input method:", ("Upload Image", "Capture from Webcam"))

    uploaded_file = None
    camera_image = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(f"Upload an image for {classifier_choice} classification...", type=["jpg", "jpeg", "png"])
    elif input_method == "Capture from Webcam":
        camera_image = st.camera_input("Take a picture")

    source_image = None
    if uploaded_file is not None:
        source_image = uploaded_file
    elif camera_image is not None:
        source_image = camera_image

    if source_image is not None:
        st.image(source_image, caption='Image for Classification', use_container_width=True)
        st.write("")
        
        # Display "Classifying..." while processing
        status_placeholder = st.empty()
        status_placeholder.write("Classifying...")

        model_to_use = None
        class_names_to_use = []
        details_to_use = {}
        csv_file_path = None

        if classifier_choice == "Butterfly Species":
            model_to_use = butterfly_species_model
            class_names_to_use = butterfly_species_names
            details_to_use = butterfly_species_info
            csv_file_path = os.path.join(DATA_DIR, "butterfly_table_species.csv")
        elif classifier_choice == "Life Stages":
            model_to_use = lifestages_model
            class_names_to_use = lifestages_names
            details_to_use = lifestages_info
            csv_file_path = os.path.join(DATA_DIR, "stages_records.csv")
        elif classifier_choice == "Pupae Defects":
            model_to_use = pupaedefects_model
            class_names_to_use = pupaedefects_names
            details_to_use = pupaedefects_info
            csv_file_path = os.path.join(DATA_DIR, "pupae_defects_quality_info.csv")
        elif classifier_choice == "Larval Diseases":
            model_to_use = larvaldiseases_model
            class_names_to_use = larvaldiseases_names
            details_to_use = larvaldiseases_info
            csv_file_path = os.path.join(DATA_DIR, "larval_diseases_quality_records.csv")

        if model_to_use:
            classification_result = classify_image(source_image, model_to_use, class_names_to_use, details_to_use)
            
            # Change the status message to "Done!" after classification
            status_placeholder.write("Done!")

            if classification_result and classification_result['class_name'] != "Unknown Class (Index out of bounds)":
                st.success(f"**{classifier_choice} Prediction:** **{classification_result['class_name']}** (Confidence: {classification_result['score']:.2f}%)")

                # Display additional information if available
                if 'scientific_name' in classification_result:
                    st.write(f"Scientific Name: {classification_result['scientific_name']}")
                    st.write(f"Family: {classification_result['family']}")
                    st.write(f"Discovered by: {classification_result['discovered']}")
                    st.write(f"Year: {classification_result['year']}")

                if 'stages_info' in classification_result:
                    st.write(f"Stages Info: {classification_result['stages_info']}")
                if 'quality_info' in classification_result:
                    st.write(f"Quality Info: {classification_result['quality_info']}")
                if 'treatment_info' in classification_result:
                    st.write(f"Treatment Info: {classification_result['treatment_info']}")

                # --- Data Logging ---
                date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Assuming Manila, Metro Manila, Philippines as current location.
                location = "Manila, Metro Manila, Philippines" 

                # Determine image path for logging
                image_filename_for_log = ""
                if uploaded_file is not None:
                    image_filename_for_log = uploaded_file.name
                elif camera_image is not None:
                    # Streamlit's camera_input provides a file-like object,
                    # but typically doesn't have a readily accessible original filename.
                    # We can use a timestamped name.
                    image_filename_for_log = f"webcam_capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    # You might want to save the actual image if needed for later review
                    # with open(os.path.join(DATA_DIR, image_filename_for_log), "wb") as f:
                    #     f.write(camera_image.getbuffer())

                log_entry = {
                    "class_name": classification_result['class_name'],
                    "score": classification_result['score'],
                    "date_time": date_time,
                    "location": location,
                    "image_path": image_filename_for_log,
                }
                # Add specific details to the log entry
                if 'scientific_name' in classification_result:
                    log_entry["scientific_name"] = classification_result['scientific_name']
                    log_entry["family"] = classification_result['family']
                    log_entry["discovered"] = classification_result["discovered"]
                    log_entry["year"] = classification_result["year"]

                if 'stages_info' in classification_result:
                    log_entry["stages_info"] = classification_result['stages_info']
                if 'quality_info' in classification_result:
                    log_entry["quality_info"] = classification_result['quality_info']
                if 'treatment_info' in classification_result:
                    log_entry["treatment_info"] = classification_result['treatment_info']

                df_log = pd.DataFrame([log_entry])

                # Append to CSV (or create if it doesn't exist)
                if os.path.exists(csv_file_path):
                    df_log.to_csv(csv_file_path, mode='a', header=False, index=False)
                else:
                    df_log.to_csv(csv_file_path, mode='w', header=True, index=False)
                
                st.success(f"Classification result logged to {csv_file_path}")

            else:
                st.error(f"Could not classify the image for {classifier_choice} or prediction was 'Unknown Class'.")
        else:
            st.warning(f"The {classifier_choice} model is not loaded. Please check the model path and ensure the model file exists.")

st.info("""

""")