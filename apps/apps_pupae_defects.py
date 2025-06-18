# -*- coding: utf-8 -*-
"""
Streamlit app for Pupae Defect Classification
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- Configuration ---
IMG_SIZE = 224
MODEL_PATH = './model/defects_app.h5' # Make sure this path is correct for your environment

# --- Load Model ---
@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_defect_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_defect_model()

# --- Class Names (ensure this matches your training data) ---
CLASS_NAMES = ['Ant bites', 'Deformed body', 'Healthy Pupae', 'Old Pupa', 'Overbend', 'Stretch abdomen']

# --- Prediction Function ---
def predict_defect(img):
    img = img.resize((IMG_SIZE, IMG_SIZE)) # Resize to model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(predictions)

    return predicted_class_name, confidence

# --- Streamlit App Layout ---
#st.set_page_config(page_title="Pupae Defect Classifier", layout="centered")

st.title("🦋 Pupae Defect Classification App")
st.write("Upload an image of a pupa to classify its defect status.")

st.sidebar.header("About")
st.sidebar.info(
    "This application uses a Convolutional Neural Network (CNN) to classify pupae images "
    "into different categories, including healthy and various defect types. "
    "The model was trained on the 'Pupae_defects' dataset."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_container_width=True)
        st.write("")
        st.write("Classifying...")

        # Perform prediction
        predicted_class, confidence = predict_defect(img)

        st.success(f"Prediction: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}**")

    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.info("Please upload a valid image file (JPG, JPEG, PNG).")

st.markdown("---")
st.markdown("Developed with ❤️ by BilaBila") # You can customize this