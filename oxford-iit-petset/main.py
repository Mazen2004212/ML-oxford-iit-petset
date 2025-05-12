from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image
import tensorflow_datasets as tfds
import tensorflow as tf

import numpy as np
import joblib
import uuid
import os
from PIL import Image
import io
import uvicorn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ========== إعداد FastAPI ==========
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

# ========== إعداد النماذج ==========
IMG_SIZE = 224
class_names = ["Bengal", "sphynx", "Cocker_Spaniel", "Great_Pyrenees", "Miniature_Pinscher"]

log_model = joblib.load("models/logistic_model.pkl")
knn_model = joblib.load("models/knn_model.pkl")
kmeans_model = joblib.load("models/kmeans_model.pkl")
cluster_labels = np.load("models/kmeans_cluster_labels.npy")
cnn_model = load_model("models/cnn_model.h5")

feature_extractor = MobileNetV2(include_top=False, pooling='avg',
                                input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form(...)):
    extension = file.filename.split(".")[-1].lower()
    if extension not in ["jpg", "jpeg", "png"]:
        return {"error": "Unsupported file type. Please upload JPG, JPEG, or PNG."}

    unique_filename = f"{uuid.uuid4()}.{extension}"
    save_path = os.path.join("uploads", unique_filename)
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    img = Image.open(save_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if model == "logistic":
        features = feature_extractor.predict(img_array)
        prediction = log_model.predict(features)
    elif model == "cnn":
        prediction = np.argmax(cnn_model.predict(img_array), axis=1)
    elif model == "knn":
        features = feature_extractor.predict(img_array)
        prediction = knn_model.predict(features)
    elif model == "kmeans":
        features = feature_extractor.predict(img_array)
        cluster = kmeans_model.predict(features)
        prediction = [int(cluster_labels[int(cluster[0])])]
    else:
        prediction = [-1]

    return {
        "prediction": class_names[int(prediction[0])],
        "image_url": f"/uploads/{unique_filename}"
    }

@app.get("/model-accuracies")
def get_accuracies():
    selected_classes = tf.constant([5, 33, 12, 15, 21], dtype=tf.int64)
    ds, _ = tfds.load('oxford_iiit_pet', split='train+test', as_supervised=True, with_info=True)

    def filter_classes(image, label):
        return tf.reduce_any(tf.equal(label, selected_classes))

    def relabel(image, label):
        idx = tf.cast(tf.where(tf.equal(selected_classes, label))[0][0], tf.int64)
        return image, idx

    def format_image(image, label):
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
        return image, label

    filtered_ds = ds.filter(filter_classes).map(relabel).map(format_image)
    images, labels = [], []
    for img, lbl in tfds.as_numpy(filtered_ds):
        images.append(img)
        labels.append(lbl)

    X = np.array(images)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_test_features = feature_extractor.predict(X_test)

    y_pred_log = log_model.predict(X_test_features)
    acc_log = accuracy_score(y_test, y_pred_log)

    y_pred_knn = knn_model.predict(X_test_features)
    acc_knn = accuracy_score(y_test, y_pred_knn)

    kmeans_preds = kmeans_model.predict(X_test_features)
    mapped_preds = [int(cluster_labels[c]) for c in kmeans_preds]
    acc_kmeans = accuracy_score(y_test, mapped_preds)

    acc_cnn = cnn_model.evaluate(X_test, y_test, verbose=0)[1]

    return {
        "Logistic Regression": float(acc_log),
        "KMeans": float(acc_kmeans),
        "KNN": float(acc_knn),
        "CNN": float(acc_cnn)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
