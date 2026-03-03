import os
import pandas as pd
from preprocessing.audio_loader import load_audio
from preprocessing.normalization import normalize_audio
from preprocessing.silence_removal import remove_silence
from features.acoustic_features import extract_features

DATA_PATH = "data/raw"

dataset = []

for label in ["healthy", "pathological"]:
    folder_path = os.path.join(DATA_PATH, label)
    
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            
            audio, sr = load_audio(file_path)
            audio = normalize_audio(audio)
            audio = remove_silence(audio, sr)
            
            features = extract_features(audio, sr)
            
            row = list(features["mfcc_mean"])
            row.append(features["pitch_mean"])
            row.append(features["zcr_mean"])
            row.append(features["centroid_mean"])
            row.append(label)
            
            dataset.append(row)

# Create column names
columns = [f"mfcc_{i}" for i in range(13)]
columns += ["pitch_mean", "zcr_mean", "centroid_mean", "label"]

df = pd.DataFrame(dataset, columns=columns)

df.to_csv("data/feature_dataset.csv", index=False)

print("Dataset created successfully.")
print("Total samples:", len(df))
print(df.head())