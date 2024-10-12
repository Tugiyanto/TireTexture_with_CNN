# app.py

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load the trained CNN model
model = load_model('model_cnn.h5')

# Streamlit app title
st.title("Aplikasi Klasifikasi Gambar dengan CNN")

# Upload image section
uploaded_file = st.file_uploader("Upload gambar untuk diklasifikasi", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded image to an array
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diupload', use_column_width=True)
    
    # Preprocess the image to fit the model input size
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (28, 28))  # Resize sesuai input model
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape to fit the model input
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    
    # Display prediction
    st.write(f"Prediksi: {predicted_class}")
