import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score

# Load RF data
df = pd.read_csv("data/feature_dataset.csv")
X = df.drop("label", axis=1)
y = df["label"].map({"healthy":0, "pathological":1})

rf_model = joblib.load("data/random_forest_model.pkl")
rf_probs = rf_model.predict_proba(X)[:, 1]

# Load CNN model
cnn_model = tf.keras.models.load_model("data/cnn_model.h5")

# Generate CNN probabilities
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

data_flow = datagen.flow_from_directory(
    "data/spectrograms",
    target_size=(224,224),
    batch_size=8,
    class_mode="binary",
    shuffle=False
)

cnn_probs = cnn_model.predict(data_flow).flatten()

# Align sizes
min_len = min(len(rf_probs), len(cnn_probs))
rf_probs = rf_probs[:min_len]
cnn_probs = cnn_probs[:min_len]
y = y[:min_len]

# Fusion
final_probs = 0.6 * rf_probs + 0.4 * cnn_probs
final_preds = (final_probs > 0.5).astype(int)

print("Fusion Accuracy:", accuracy_score(y, final_preds))
print("Fusion AUC:", roc_auc_score(y, final_probs))