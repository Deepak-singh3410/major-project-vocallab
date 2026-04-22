import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import matplotlib.pyplot as plt

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model("data/multiclass_cnn.keras")

CLASS_NAMES = ["Laryngozele", "Normal", "Parkinson", "Vox_senilis"]

# -----------------------------
# RECORD AUDIO
# -----------------------------
def record_audio(duration=5, fs=44100):
    st.info("Recording... Speak now")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    st.success("Recording complete")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, fs, audio)

    return temp_file.name

# -----------------------------
# GENERATE SPECTROGRAM
# -----------------------------
def generate_spectrogram(audio_path):
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

    # Save image
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
    plt.axis('off')

    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_img.name, bbox_inches='tight', pad_inches=0)
    plt.close()

    return temp_img.name

# -----------------------------
# PREDICT
# -----------------------------
def predict(audio_path):
    spec_path = generate_spectrogram(audio_path)

    img = tf.keras.preprocessing.image.load_img(spec_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_index = np.argmax(pred)

    return CLASS_NAMES[class_index], np.max(pred)

# -----------------------------
# UI
# -----------------------------
st.title("🎤 Vocal Disease Detection System")

st.write("Click below and speak for 5 seconds")

if st.button("🎙 Record Voice"):
    audio_path = record_audio()

    st.audio(audio_path)

    st.write("Processing...")

    label, confidence = predict(audio_path)

    st.success(f"Diagnosis: {label}")
    st.info(f"Confidence: {confidence:.2f}")