import librosa
import numpy as np

def remove_silence(audio, sr, top_db=20):
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return audio
    return np.concatenate([audio[start:end] for start, end in intervals])