import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request

from keras.applications.resnet50 import preprocess_input

MODEL_URL = (
    "https://huggingface.co/Alphonsus01/best_resnet50_fold5/resolve/main/"
    "best_resnet50_fold5.keras"
)

st.set_page_config(
    page_title="Klasifikasi Varietas Mangga",
    page_icon="🥭",
    layout="centered"
)

st.title("🥭 Klasifikasi Varietas Mangga")
st.write("Upload gambar daun mangga untuk memprediksi varietas mangga.")
st.write("Khusus untuk varietas mangga Cokanan, Erwin, Gedong Gincu, Harum Manis, dan Mahatir.")

MODEL_PATH = "best_resnet50_fold5.keras"

CLASS_NAMES = [
    "Cokanan",
    "Erwin",
    "Gedong Gincu",
    "Harum Manis",
    "Mahatir"
]

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh model dari Hugging Face..."):
            urllib.request.urlretrieve(
                MODEL_URL,
                MODEL_PATH
            )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

try:
    model = load_model()
    st.info("Klasifikasi dilakukan dengan menggunakan model ResNet50 yang telah dilatih.")
except Exception as e:
    st.error(f"Gagal memuat model:\n\n{e}")
    st.stop()


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image).astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    image_array = preprocess_input(image_array)

    return image_array


uploaded_file = st.file_uploader(
    "Pilih gambar mangga",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Gambar yang diupload",
        use_container_width=True
    )

    processed_image = preprocess_image(image)

    with st.spinner("Melakukan prediksi..."):

        prediction = model.predict(processed_image)

        predicted_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = np.max(prediction) * 100

    st.subheader("📊 Hasil Prediksi")

    st.success(f"Varietas Mangga: **{predicted_class}**")

    st.info(f"Tingkat Keyakinan: **{confidence:.2f}%**")

    st.subheader("📈 Probabilitas Semua Kelas")

    for i, class_name in enumerate(CLASS_NAMES):
        prob = prediction[0][i] * 100
        st.write(f"{class_name}: {prob:.2f}%")
        st.progress(float(prob) / 100)