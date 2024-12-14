import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Fungsi untuk memuat model (pastikan Anda sudah menyimpan model Anda)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cnn_model.keras')  # Ganti dengan path model Anda
    return model

def predict_texture(image, model):
    # Preprocessing gambar sesuai kebutuhan model Anda
    image = image.resize((224, 224))  # Ganti ukuran sesuai dengan input model Anda
    image_array = np.array(image) / 255.0  # Normalisasi
    image_array = np.expand_dims(image_array, axis=0)  # Tambahkan dimensi batch

    predictions = model.predict(image_array)
    return predictions

# Muat model
model = load_model()

# Tampilan aplikasi Streamlit
st.title("Klasifikasi Tekstur Ban")
st.write("Upload gambar ban untuk mengetahui teksturnya.")

uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Prediksi dengan model
    predictions = predict_texture(image, model)

    # Ambil label prediksi (pastikan label sesuai dengan yang ada di model Anda)
    class_names = ['normal', 'cracked']  # Ganti dengan kelas yang sesuai
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f"Prediksi: **{predicted_class}**")
    st.write(f"Probabilitas: {predictions[0][np.argmax(predictions)]:.2f}")

st.write("\n\nMade with ❤️ using Streamlit")
