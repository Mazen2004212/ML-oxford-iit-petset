import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image


IMG_SIZE = 224
class_names = ["Bengal", "Sphynx", "Cocker_Spaniel", "Great_Pyrenees", "Miniature_Pinscher"]

# ========== download ==========
log_model = joblib.load("logistic_model.pkl")
cnn_model = load_model("cnn_model.h5")
knn_model = joblib.load("knn_model.pkl")
feature_extractor = MobileNetV2(include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')

# ========== التصنيف ==========
def classify_image(img_path, model_choice):
    img = keras_image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
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
    else:
        prediction = [-1]

    return class_names[int(prediction[0])]

# ========== وظائف الواجهة ==========
def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).resize((200, 200))
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo)
        image_label.image = photo
        selected_image_path.set(file_path)
        result_label.config(text="")

def run_prediction():
    path = selected_image_path.get()
    model_choice = model_var.get()
    if path and model_choice:
        result = classify_image(path, model_choice)
        result_label.config(text=f"Prediction: {result}", fg="green")

# ========== إنشاء الواجهة ==========
window = tk.Tk()
window.title("Pet Breed Classifier")
window.geometry("420x500")
window.configure(bg="#f9f9f9")

tk.Label(window, text="Select Model:", bg="#f9f9f9", font=("Arial", 12)).pack(pady=10)

model_var = tk.StringVar(value="Logistic Regression")
model_menu = ttk.Combobox(window, textvariable=model_var, values=["Logistic Regression", "CNN", "KNN"], state="readonly")
model_menu.pack()

browse_btn = tk.Button(window, text="Browse Image", command=browse_image, font=("Arial", 11), bg="#4CAF50", fg="white")
browse_btn.pack(pady=10)

image_label = tk.Label(window, bg="#ddd", width=200, height=200)
image_label.pack(pady=5)

selected_image_path = tk.StringVar()

predict_btn = tk.Button(window, text="Predict", command=run_prediction, font=("Arial", 11), bg="#2196F3", fg="white")
predict_btn.pack(pady=15)

result_label = tk.Label(window, text="Prediction: ", font=("Arial", 14), bg="#f9f9f9")
result_label.pack(pady=10)

window.mainloop()
