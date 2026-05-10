import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# OPTIONAL RANDOM FOREST IMPORT
try:
    import joblib
    RF_AVAILABLE = True
except:
    RF_AVAILABLE = False

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_DIR = os.path.join(BASE_DIR, "data")

# -----------------------------
# CLASS NAMES
# -----------------------------
SPEC_DIR = os.path.join(DATA_DIR, "spectrograms_multiclass")

if os.path.isdir(SPEC_DIR):

    CLASS_NAMES = sorted([
        d for d in os.listdir(SPEC_DIR)
        if os.path.isdir(os.path.join(SPEC_DIR, d))
    ])

else:

    CLASS_NAMES = [
        "Laryngozele",
        "Normal",
        "Parkinsons",
        "Vox_senilis"
    ]

# -----------------------------
# LOAD CNN MODEL
# -----------------------------
cnn_path = os.path.join(
    DATA_DIR,
    "multiclass_cnn.keras"
)

if not os.path.exists(cnn_path):

    cnn_path = os.path.join(
        DATA_DIR,
        "cnn_model.h5"
    )

cnn_model = tf.keras.models.load_model(cnn_path)

# -----------------------------
# OPTIONAL RANDOM FOREST MODEL
# -----------------------------
rf_model = None

RF_PATH = os.path.join(
    DATA_DIR,
    "random_forest_model.pkl"
)

if RF_AVAILABLE and os.path.exists(RF_PATH):

    try:
        rf_model = joblib.load(RF_PATH)
        print("✅ Random Forest model loaded")

    except Exception as e:
        print("❌ RF model load failed:", e)

# -----------------------------
# IMAGE SIZE
# -----------------------------
IMG_SIZE = (224, 224)

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_rf_features(audio_path):

    y, sr = librosa.load(audio_path, sr=44100)

    # MFCC
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_means = np.mean(mfccs, axis=1)

    # Pitch
    pitch = librosa.yin(
        y,
        fmin=50,
        fmax=500
    )

    pitch_mean = np.mean(pitch)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

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
def generate_spectrogram(
    audio_path,
    output_path="temp_spec.png"
):

    y, sr = librosa.load(audio_path, sr=44100)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        fmax=8000
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # SAFE NORMALIZATION
    mel_db -= mel_db.min()

    if mel_db.max() != 0:
        mel_db /= mel_db.max()

    mel_db *= 255

    import matplotlib.pyplot as plt

    plt.figure(figsize=(2.24, 2.24), dpi=100)

    plt.imshow(
        mel_db,
        aspect='auto',
        origin='lower',
        cmap='magma'
    )

    plt.axis('off')

    plt.savefig(
        output_path,
        bbox_inches='tight',
        pad_inches=0
    )

    plt.close()

    return output_path

# -----------------------------
# PREDICTION
# -----------------------------
def predict(audio_path):

    # -----------------------------
    # OPTIONAL RANDOM FOREST STAGE
    # -----------------------------
    if rf_model is not None:

        try:

            features = extract_rf_features(audio_path)

            rf_prediction = rf_model.predict(features)[0]

            # Healthy prediction
            if rf_prediction == 0:

                try:
                    prob = float(
                        rf_model.predict_proba(features)[0][0]
                    )

                except:
                    prob = 0.90

                return ("Normal", prob)

        except Exception as e:

            print("RF Prediction Error:", e)

    # -----------------------------
    # CNN MULTICLASS PREDICTION
    # -----------------------------
    spec_path = generate_spectrogram(audio_path)

    img = load_img(
        spec_path,
        target_size=IMG_SIZE
    )

    img = img_to_array(img)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = cnn_model.predict(img)

    class_index = int(
        np.argmax(prediction, axis=1)[0]
    )

    confidence = float(
        np.max(prediction)
    )

    raw_label = CLASS_NAMES[class_index]

    # -----------------------------
    # CLEAN LABELS
    # -----------------------------
    display_map = {
        "Vox_senilis": "Vox Senilis",
        "Vox senilis": "Vox Senilis",
        "Parkinsions": "Parkinsons",
        "Parkinsons": "Parkinsons",
        "Normal": "Normal",
        "Laryngozele": "Laryngozele"
    }

    final_label = display_map.get(
        raw_label,
        raw_label.replace("_", " ")
    )

    return (final_label, confidence)

# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":

    sample = "data/raw/sample.wav"

    if os.path.exists(sample):

        result = predict(sample)

        print("\nFinal Diagnosis:", result)

    else:

        print("Sample file not found.")