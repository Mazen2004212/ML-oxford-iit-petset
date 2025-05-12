import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image
import os

# ========== constants ==========
IMG_SIZE = 224
class_names = ["Bengal", "sphynx", "Cocker_Spaniel", "Great_Pyrenees", "Miniature_Pinscher"]

# ========== load models ==========
log_model = joblib.load("logistic_model.pkl")
cnn_model = load_model("cnn_model.h5")
knn_model = joblib.load("knn_model.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")
cluster_labels = np.load("kmeans_cluster_labels.npy")
feature_extractor = MobileNetV2(include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')

# ========== classify ==========
def classify_image(img, model_choice):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if model_choice == "Logistic Regression":
        features = feature_extractor.predict(img_array)
        prediction = log_model.predict(features)

    elif model_choice == "CNN":
        prediction = np.argmax(cnn_model.predict(img_array), axis=1)

    elif model_choice == "KNN":
        features = feature_extractor.predict(img_array)
        prediction = knn_model.predict(features)

    elif model_choice == "KMeans":
        features = feature_extractor.predict(img_array)
        cluster = kmeans_model.predict(features)
        prediction = [int(cluster_labels[int(cluster[0])])]

    else:
        prediction = [-1]

    return class_names[int(prediction[0])]

# ========== streamlit ==========
st.set_page_config(page_title="Pet Breed Classifier", layout="centered")
st.title("🐾 Pet Breed Classifier")

model_choice = st.selectbox("Select Model", ["Logistic Regression", "CNN", "KNN", "KMeans"])
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        prediction = classify_image(img, model_choice)
        st.success(f"Prediction: {prediction}")

    if st.button("Clear"):
        st.session_state.clear()
        st.experimental_rerun()