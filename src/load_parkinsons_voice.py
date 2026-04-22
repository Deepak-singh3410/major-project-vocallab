from datasets import load_dataset
import soundfile as sf
import os

print("Loading dataset (NO TORCHCODEC)...")

ds = load_dataset(
    "birgermoell/Italian_Parkinsons_Voice_and_Speech",
    streaming=True
)

output_dir = "data/raw_multiclass/Parkinson"
os.makedirs(output_dir, exist_ok=True)

count = 0

for sample in ds["train"]:
    try:
        audio = sample["audio"]

        # 🔥 SAFE extraction (no torchcodec)
        array = audio["array"]
        sr = audio["sampling_rate"]

        file_path = f"{output_dir}/parkinson_{count}.wav"
        sf.write(file_path, array, sr)

        print(f"Saved: {file_path}")
        count += 1

        if count >= 100:  # limit for now
            break

    except Exception as e:
        print("Skipping error:", e)

print("Done saving Parkinson audio!")