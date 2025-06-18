import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
import datetime  # Import datetime for date handling


st.header('Butterfly Larval Disease Classification')

# Define disease names and treatment information
# Ensure these lists are aligned by index
larvaldiseases_names = ['Anaphylaxis Infection',
                        'Gnathostomiasis',
                        'Healthy Larvae',
                        'Nucleopolyhedrosis']
treatment_info_details = [
    "Prevention for Anaphylaxis Infection is to provide a ground container with water or gasoline that holds the feet of the cagenet to create an impenetrable barrier for ants.\n This is non-toxic method to protect larvae from ant predation or irritation by creating a physical moat or barrier around their enclosure",
    "Prevention for Gnathostomiasis is to keep away from the dark area where they live.",
    "Healthy Larvae! There is no disease on the larvae",
    "Prevention for Nucleopolyhedrosis removed the larvae who have injected. These viruses infect the larvae, leading to symptoms such as tissue liquefaction and eventual death. Baculoviruses are often used as biological control agents in agriculture to manage pest populations."
]


# Load the Keras model
# It's good practice to wrap model loading in a try-except block for robustness
try:
    model = load_model('./model/model_Larval_Diseases.h5')
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load the model. Error: {e}")
    model = None # Set model to None if loading fails to prevent further errors


def classify_images(image_path):
    if model is None:
        st.error("Model not loaded, cannot perform classification.")
        return None
    try:
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        predicted_class_index = np.argmax(result)
        predicted_class_name = larvaldiseases_names[predicted_class_index]
        predicted_score = np.max(result) * 100

        # Correctly get the treatment information using the predicted index
        predicted_treatment_info = treatment_info_details[predicted_class_index]

        outcome = {
            "class_name": predicted_class_name,
            "score": predicted_score,
            "treatment_info": predicted_treatment_info, # This now holds the detailed treatment
        }
        return outcome
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None


uploaded_file = st.file_uploader('Upload an Image of a Butterfly Larva')
if uploaded_file is not None:
    try:
        # Create the 'upload' directory if it doesn't exist
        upload_dir = 'upload'
        os.makedirs(upload_dir, exist_ok=True)

        # Save the uploaded file
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Display the uploaded image with a div for styling
        st.markdown(
            f'<div class="border"><img src="data:image/jpeg;base64,{uploaded_file.getvalue().hex()}" width="200" caption="Uploaded Image"></div>',
            unsafe_allow_html=True
        )
        st.image(uploaded_file, width=200, caption="Uploaded Image")


        # Classify the image
        classification_result = classify_images(file_path)
        if classification_result:
            st.markdown(
                f"**Predicted Disease:** **{classification_result['class_name']}** "
                f"with a confidence score of **{classification_result['score']:.2f}%**."
            )
            st.markdown(f"**Treatment Information:** {classification_result['treatment_info']}")

            # Get additional information
            date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            location = "Marinduque, Philippines"  # Replace with actual location data if available

            # Create a DataFrame for the result
            disease_data = {
                "disease_name": [classification_result['class_name']], # Use 'class_name'
                "treatment_info": [classification_result['treatment_info']],
                "score": [classification_result['score']],
                "date_time": [date_time],
                "location": [location],
                "image_path": [uploaded_file.name],  # Store filename
            }
            df_species = pd.DataFrame(disease_data)

            # Append to CSV (or create if it doesn't exist)
            csv_file = "./data/larval_disease_records.csv" # Renamed for clarity
            data_dir = './data'
            os.makedirs(data_dir, exist_ok=True) # Ensure 'data' directory exists

            if os.path.exists(csv_file):
                df_existing = pd.read_csv(csv_file)
                df_combined = pd.concat([df_existing, df_species], ignore_index=True)
                df_combined.to_csv(csv_file, index=False)
            else:
                df_species.to_csv(csv_file, index=False)
            st.success(f"Classification data saved to {csv_file}")

        else:
            st.error("Classification failed. Please try another image.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Add the CSS styling
st.markdown(
    """
    <style>
        * {
            margin: 0px;
            padding: 0px;
            box-sizing: border-box;
        }

        .border img {
            border-radius: 15px;
            border: 2px solid black;
        }
    </style>
    """,
    unsafe_allow_html=True
)