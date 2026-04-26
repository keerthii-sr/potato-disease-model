import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("potato_disease_model.h5")

classes = ['Early Blight', 'Late Blight', 'Healthy']

def predict(image):
    img = cv2.resize(image, (128,128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    return classes[np.argmax(pred)]

st.title("🥔 Potato Disease Detection")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = np.array(image)
    result = predict(img)

    st.success(f"Prediction: {result}")