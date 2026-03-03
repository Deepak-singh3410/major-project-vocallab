import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "data/raw_multiclass"
OUTPUT_PATH = "data/spectrograms_multiclass"

classes = ["Normal", "Laryngozele", "Vox_senilis"]

for label in classes:
    input_folder = os.path.join(DATA_PATH, label)
    output_folder = os.path.join(OUTPUT_PATH, label)
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.lower().endswith(".wav"):
            file_path = os.path.join(input_folder, file)

            y, sr = librosa.load(file_path, sr=44100)

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

            plt.figure(figsize=(2.56, 5.12), dpi=100)
            plt.imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
            plt.axis('off')

            output_file = os.path.join(output_folder, file.replace(".wav", ".png"))
            plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
            plt.close()

print("Multiclass spectrogram dataset created.")