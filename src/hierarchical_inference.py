import os
import numpy as np
import joblib
import librosa
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -----------------------------
# PATHS & LOAD MODELS
# -----------------------------
# Make data paths robust relative to this file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

rf_model = joblib.load(os.path.join(DATA_DIR, "random_forest_model.pkl"))
# Derive class names from the training spectrogram folder to ensure ordering
SPEC_DIR = os.path.join(DATA_DIR, "spectrograms_multiclass")
if os.path.isdir(SPEC_DIR):
    CLASS_NAMES = sorted([
        d for d in os.listdir(SPEC_DIR)
        if os.path.isdir(os.path.join(SPEC_DIR, d))
    ])
else:
    # fallback to a safe default
    CLASS_NAMES = ["Laryngozele", "Normal", "Vox_senilis", "Parkinsions"]

# prefer the explicit h5 if present, otherwise fallback to saved keras folder
cnn_path = os.path.join(DATA_DIR, "cnn_model.h5")
if not os.path.exists(cnn_path):
    cnn_path = os.path.join(DATA_DIR, "multiclass_cnn.keras")


cnn_model = tf.keras.models.load_model(cnn_path)

# If the loaded model's output doesn't match the number of class folders,
# prefer the explicit multiclass model file if available.
try:
    out_neurons = int(cnn_model.output_shape[-1])
except Exception:
    out_neurons = None

if out_neurons is None or os.path.isdir(SPEC_DIR) and out_neurons != len([d for d in os.listdir(SPEC_DIR) if os.path.isdir(os.path.join(SPEC_DIR, d))]):
    alt_path = os.path.join(DATA_DIR, "multiclass_cnn.keras")
    if os.path.exists(alt_path) and alt_path != cnn_path:
        cnn_model = tf.keras.models.load_model(alt_path)
        try:
            out_neurons = int(cnn_model.output_shape[-1])
        except Exception:
            out_neurons = None

# Derive class names from the training spectrogram folder to ensure ordering
SPEC_DIR = os.path.join(DATA_DIR, "spectrograms_multiclass")
IMG_SIZE = (224, 224)

# -----------------------------
# FEATURE EXTRACTION (MATCH TRAINING)
# -----------------------------
def extract_rf_features(audio_path):
    y, sr = librosa.load(audio_path, sr=44100)

    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)

    # Pitch
    pitch = librosa.yin(y, fmin=50, fmax=500)
    pitch_mean = np.mean(pitch)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)

    features = np.hstack([
        mfcc_means,
        pitch_mean,
        zcr_mean,
        centroid_mean
    ])

    return features.reshape(1, -1)

# -----------------------------
# GENERATE SPECTROGRAM
# -----------------------------
def generate_spectrogram(audio_path, output_path="temp_spec.png"):
    y, sr = librosa.load(audio_path, sr=44100)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        fmax=8000
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db -= mel_db.min()
    mel_db /= mel_db.max()
    mel_db *= 255

    import matplotlib.pyplot as plt

    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return output_path

# -----------------------------
# HIERARCHICAL PREDICTION
# -----------------------------
def predict(audio_path):

    # -------- Step 1: Binary RF --------
    features = extract_rf_features(audio_path)
    rf_prediction = rf_model.predict(features)[0]

    # If RF says healthy (0) -> skip CNN and return Normal with RF probability
    if rf_prediction == 0:
        try:
            prob = float(rf_model.predict_proba(features)[0][0])
        except Exception:
            prob = 0.9
        return ("Normal", prob)

    # -------- Step 2: Multiclass CNN --------
    spec_path = generate_spectrogram(audio_path)

    img = load_img(spec_path, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    cnn_prediction = cnn_model.predict(img)
    class_index = int(np.argmax(cnn_prediction, axis=1)[0])
    prob = float(np.max(cnn_prediction))

    raw_label = CLASS_NAMES[class_index]

    # map common dataset names to user-friendly labels
    display_map = {
        "Vox_senilis": "Vox Senilis",
        "Vox senilis": "Vox Senilis",
        "Parkinsions": "Parkinsons",
        "Parkinsons": "Parkinsons",
        "Normal": "Normal",
        "Laryngozele": "Laryngozele"
    }

    display_label = display_map.get(raw_label, raw_label.replace("_", " "))

    return (display_label, prob)

# -----------------------------
# TEST RUN
# -----------------------------
if __name__ == "__main__":
    test_audio = "data/raw/sample.wav"  # change if needed
    result = predict(test_audio)
    print("Final Diagnosis:", result)

# -----------------------------
# TEST MULTIPLE FILES
# -----------------------------
if __name__ == "__main__":

    test_files = {
        "Normal Test": r"archive/patient-vocal-dataset-small/patient-vocal-dataset-small/Normal/4-phrase.wav",
        "Laryngozele Test": r"archive/patient-vocal-dataset-small/patient-vocal-dataset-small/Laryngozele/1205-phrase.wav",
        "Vox Senilis Test": r"archive/patient-vocal-dataset-small/patient-vocal-dataset-small/Vox senilis/816-phrase.wav"
    }

    print("\n===== HIERARCHICAL SYSTEM TESTING =====\n")

    for label, path in test_files.items():

        if not os.path.exists(path):
            print(f"{label}: File not found -> {path}")
            continue

        result = predict(path)

        print(f"{label}")
        print(f"File: {path}")
        print(f"Predicted Diagnosis: {result}")
        print("-" * 50)