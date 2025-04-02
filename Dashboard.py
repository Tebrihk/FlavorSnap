import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("C:/Users/tebrick_KING/FlavorSnap/model.keras")

# Class labels (modify based on your dataset)
class_names = ["Akara", "Bread", "Egusi", "Moi Moi", "Rice and Stew", "Yam"]

st.title("🍔 FlavorSnap - Food Recognition")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f"### 🏷️ Identified as: **{predicted_class}**")