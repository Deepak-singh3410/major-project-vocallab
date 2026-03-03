import numpy as np

def normalize_audio(audio):
    max_val = np.max(abs(audio))
    if max_val == 0:
        return audio
    return audio / max_val