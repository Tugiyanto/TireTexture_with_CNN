import streamlit as st
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model

# Load the previously saved model
model = load_model('cnn_model.h5')

# Check the architecture of the model
model.summary()


# Fungsi untuk memuat dan memproses gambar
def load_and_process_image(img, target_size=(224, 224)):
    img = img.resize(target_size)  # Resize gambar sesuai input CNN
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch
    img_array /= 255.  # Normalisasi
    return img_array

# UI menggunakan Streamlit
st.title("Pengenalan Citra dengan CNN")
st.write("Unggah gambar untuk deteksi")

uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang akan diunggah", use_column_width=True)
    
    # Memproses gambar
    st.write("Memproses gambar...")
    processed_image = load_and_process_image(img)
    
    # Melakukan prediksi
    prediction = model.predict(processed_image)
    pred_class = np.argmax(prediction, axis=1)  # Kelas prediksi
    
    # Menampilkan hasil prediksi
    st.write(f"Hasil Prediksi: {pred_class[0]}")
