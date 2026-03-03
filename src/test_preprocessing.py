from preprocessing.audio_loader import load_audio
from preprocessing.normalization import normalize_audio
from preprocessing.silence_removal import remove_silence

file_path = "data/raw/Sample1.wav"

audio, sr = load_audio(file_path)
audio = normalize_audio(audio)
audio = remove_silence(audio, sr)

print("Sampling Rate:", sr)
print("Processed Audio Length:", len(audio))
print("Processed Audio:", audio)