# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("cifar10_cnn_model.keras")

# CIFAR-10 classes
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Streamlit page config
st.set_page_config(page_title="CIFAR-10 Image Classification", page_icon="📷")
st.title("📷 Image Classification App")
st.write("Upload an image and the model will predict its class.")

# Sidebar info
st.sidebar.header("About the Project")
st.sidebar.info("""
**Created by:** Mirza Yasir Abdullah Baig  
- [LinkedIn](https://www.linkedin.com/in/mirza-yasir-abdullah-baig/)  
- [Kaggle](https://www.kaggle.com/code/mirzayasirabdullah07)  
- [GitHub](https://github.com/mirzayasirabdullahbaig07)  
This project classifies images into 10 CIFAR-10 categories.
""")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess image
    img = image.resize((32, 32))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape (1, 32, 32, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    st.subheader("Prediction")
    st.write(f"The uploaded image is classified as: **{predicted_class}**")
