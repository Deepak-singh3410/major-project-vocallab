import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "data/raw_multiclass"
OUTPUT_PATH = "data/spectrograms_multiclass"

# ✅ Get only valid class folders
classes = [
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d))
]

print("Detected classes:", classes)

for label in classes:
    input_folder = os.path.join(DATA_PATH, label)
    output_folder = os.path.join(OUTPUT_PATH, label)

    os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(input_folder)
    print(f"\nProcessing class: {label} | Files: {len(files)}")

    for file in files:
        if not file.lower().endswith(".wav"):
            continue

        try:
            input_path = os.path.join(input_folder, file)
            output_file = os.path.join(
                output_folder,
                file.replace(".wav", ".png")
            )

            # ✅ Skip if already processed
            if os.path.exists(output_file):
                continue

            # Load audio
            y, sr = librosa.load(input_path, sr=44100)

            # Generate mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=128,
                fmax=8000
            )

            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Normalize safely
            mel_db -= mel_db.min()
            if mel_db.max() != 0:
                mel_db /= mel_db.max()
            mel_db *= 255

            # Plot
            plt.figure(figsize=(2.56, 5.12), dpi=100)
            plt.imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
            plt.axis('off')

            plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
            plt.close()

        except Exception as e:
            print(f"❌ Skipping {file}: {e}")

print("\n✅ Multiclass spectrogram dataset created successfully.")