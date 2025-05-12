import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from scipy.stats import mode
import joblib
import os

# ========== [1] Load and preprocess ==========
IMG_SIZE = 224
selected_classes = tf.constant([5, 33, 12, 15, 21], dtype=tf.int64)

ds, ds_info = tfds.load('oxford_iiit_pet', split='train+test', as_supervised=True, with_info=True)
label_names = ds_info.features['label'].names
print("\n📌 Selected Classes:")
for class_id in selected_classes.numpy():
    print(f"{class_id} → {label_names[class_id]}")

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
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ========== [2] Feature Extraction ==========
print("\n📡 Extracting Features...")
base_model = MobileNetV2(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg', weights='imagenet')
X_train_features = base_model.predict(X_train)
X_test_features = base_model.predict(X_test)

# ========== [3] Logistic Regression ==========
print("\n🔎 Logistic Regression")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_features, y_train)
joblib.dump(log_model, "logistic_model.pkl")
y_pred_log = log_model.predict(X_test_features)
acc_log = np.mean(y_pred_log == y_test)
print("Accuracy:", acc_log)
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# ========== [4] KMeans Clustering ==========
print("\n🔎 KMeans Clustering")
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_train_features)
cluster_labels = np.zeros(5)
for i in range(5):
    cluster_data = y_train[kmeans.labels_ == i]
    if len(cluster_data) > 0:
        m = mode(cluster_data, keepdims=True)
        cluster_labels[i] = m.mode[0]
    else:
        cluster_labels[i] = 0
y_pred_kmeans = cluster_labels[kmeans.predict(X_test_features)]
acc_kmeans = np.mean(y_pred_kmeans == y_test)
print("Accuracy:", acc_kmeans)
print(confusion_matrix(y_test, y_pred_kmeans))
print(classification_report(y_test, y_pred_kmeans))

# ✅ Save KMeans model and cluster labels
joblib.dump(kmeans, "kmeans_model.pkl")
np.save("kmeans_cluster_labels.npy", cluster_labels)

# ========== [5] KNN Classifier ==========
print("\n🔎 KNN Classifier")
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_features, y_train)
joblib.dump(knn_model, "knn_model.pkl")
y_pred_knn = knn_model.predict(X_test_features)
acc_knn = np.mean(y_pred_knn == y_test)
print("Accuracy:", acc_knn)
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# ========== [6] CNN Classifier ==========
print("\n🔎 CNN Model")
data_augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

cnn_model = models.Sequential([
    data_augment,
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(5, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)
cnn_model.save("cnn_model.h5")

# Accuracy curve
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("CNN Accuracy Curve")
plt.legend()
plt.show()

loss, acc_cnn = cnn_model.evaluate(X_test, y_test)
y_pred_cnn = np.argmax(cnn_model.predict(X_test), axis=1)
print("Test Accuracy:", acc_cnn)
print(confusion_matrix(y_test, y_pred_cnn))
print(classification_report(y_test, y_pred_cnn))

# ========== [7] Compare Model Accuracies ==========
plt.bar(["Logistic", "KMeans", "KNN", "CNN"], [acc_log, acc_kmeans, acc_knn, acc_cnn])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# ✅ Print working directory
print("\n🗂️ Files saved in:", os.getcwd())
