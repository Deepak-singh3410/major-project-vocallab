import librosa
import numpy as np

def extract_features(audio, sr):
    features = {}

    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features["mfcc_mean"] = np.mean(mfcc, axis=1)

    # Pitch (Fundamental Frequency)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    if len(pitch_values) > 0:
        features["pitch_mean"] = np.mean(pitch_values)
    else:
        features["pitch_mean"] = 0

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features["zcr_mean"] = np.mean(zcr)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features["centroid_mean"] = np.mean(centroid)

    return features