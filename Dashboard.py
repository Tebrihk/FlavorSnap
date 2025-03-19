import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('C:/Users/tebrick_KING/FlavorSnap/model.keras')

def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    return np.expand_dims(normalized_image_array, axis=0)

st.title('Food Image Classifier')
st.write('Upload an image of a food item, and the model will predict its category.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write('Classifying...')
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)
    class_names = ['Bread', 'Rice', 'Pasta', 'Salad', 'Fruit', 'Dessert']  # Update with your actual class names
    st.write(f'Prediction: {class_names[predicted_class[0]]}')


