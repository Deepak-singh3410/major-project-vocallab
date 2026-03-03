import librosa

TARGET_SR = 44100

def load_audio(file_path, target_sr=TARGET_SR):
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr