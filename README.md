# 🐾 Oxford-IIT Pet Image Classifier

This project is a web-based image classification system for the Oxford-IIIT Pet Dataset using multiple machine learning models: Logistic Regression, K-Nearest Neighbors (KNN), KMeans Clustering, and Convolutional Neural Networks (CNN). Users can upload images of pets and classify their breed through an interactive GUI or web interface.

## 🧠 Features

- Upload and classify pet images (cats and dogs)
- Choose from four ML models: Logistic Regression, KNN, KMeans, and CNN
- Audio feedback for correct/wrong predictions
- GUI interface using Python
- Web interface using Flask and HTML/CSS

## 📁 Project Structure

oxford-iit-petset/
│
├── app.py # Flask app entry point
├── main.py # Model loader and predictor logic
├── gui/
│ ├── oxford-petset.py # GUI application
│ ├── gui_predictor.py # GUI model integration
│ └── test.py # Testing script
│
├── models/ # Trained ML models
│ ├── cnn_model.h5
│ ├── logistic_model.pkl
│ ├── knn_model.pkl
│ ├── kmeans_model.pkl
│ └── kmeans_cluster_labels.npy
│
├── static/
│ ├── style.css
│ └── sounds/ # Sound effects
│ ├── correct.mp3
│ └── wrong.mp3
│
├── templates/
│ └── index.html # Web interface layout
│
└── uploads/ # Uploaded images


## 🛠️ Installation

1. Clone the repository or extract the zip file.
2. Install dependencies:
```bash
pip install -r requirements.txt


python app.py

python gui/oxford-petset.py


Dependencies
Flask

TensorFlow / Keras

scikit-learn

NumPy

OpenCV (if GUI uses it)

matplotlib (for image display)

🧪 Models Used
Logistic Regression: Trained using feature-extracted images.

KMeans: Unsupervised model clustered on pet images.

KNN: Based on nearest neighbor classification.

CNN: Deep learning model trained on raw images.

📸 Dataset
Oxford-IIIT Pet Dataset — includes 37 categories with roughly 200 images each of cats and dogs.

✨ Contributors
Developed by: [Mazen ibrahim,Mohamed abd el-gawad,Mohamed ahmed ,Hala mazen]

Dataset: Oxford-IIIT Pet Dataset
