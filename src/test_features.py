from preprocessing.audio_loader import load_audio
from preprocessing.normalization import normalize_audio
from preprocessing.silence_removal import remove_silence
from features.acoustic_features import extract_features

file_path = "data/raw/Sample1.wav"

audio, sr = load_audio(file_path)
audio = normalize_audio(audio)
audio = remove_silence(audio, sr)

features = extract_features(audio, sr)

print("Extracted Features:")
for key, value in features.items():
    print(key, ":", value)

print("Sampling Rate:", sr)
print("Processed Audio Length:", len(audio))
print("Processed Audio:", audio)
print("Features:", features)