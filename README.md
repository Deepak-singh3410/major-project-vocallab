# 🎙️ Hierarchical Vocal Pathology Detection System

## 📌 Overview
This project presents an AI-based system for detecting vocal disorders using speech signals.  
It combines classical machine learning and deep learning in a hierarchical pipeline to improve diagnostic accuracy.

The system analyzes voice recordings and predicts whether a subject is:
- Healthy
- Laryngozele
- Vox Senilis
- Parkinson’s Disease

---

## 🧠 Key Idea (Hierarchical Model)
Instead of directly classifying all diseases, the system works in two stages:

1. **Random Forest (Binary Classification)**
   - Healthy vs Pathological

2. **CNN (Multiclass Classification)**
   - If pathological → classify into:
     - Laryngozele
     - Vox Senilis
     - Parkinson’s

This improves efficiency and reduces misclassification.

---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Librosa (audio processing)
- NumPy / Pandas
- Matplotlib

---

## 📁 Project Structure
vocal/
│
├── src/
│ ├── hierarchical_inference.py
│ ├── train_model.py
│ ├── train_multiclass_cnn.py
│ ├── train_parkinsons_model.py
│ ├── generate_spectrograms.py
│ └── features/
│ └── acoustic_features.py
│
├── data/ (excluded from repo)
├── README.md
└── requirements.txt


---

## 🔬 Methodology

### 1. Audio Preprocessing
- Load `.wav` files using Librosa
- Normalize audio
- Remove silence
- Extract acoustic features

### 2. Feature Extraction
- MFCC
- Pitch (Fo, Fhi, Flo)
- Jitter & Shimmer
- Harmonics-to-noise ratio (HNR)

### 3. Spectrogram Generation
- Convert audio → Mel Spectrogram
- Used as input for CNN

### 4. Model Training
- Random Forest → Binary classifier
- CNN → Multiclass classifier

---

## 📊 Results

| Class          | Precision | Recall | F1-score |
|----------------|----------|--------|----------|
| Laryngozele    | ~0.20    | ~0.30  | ~0.23    |
| Normal         | ~0.85    | ~0.70  | ~0.77    |
| Parkinson’s    | ~0.98    | ~0.98  | ~0.98    |
| Vox Senilis    | ~0.75    | ~0.80  | ~0.78    |

**Overall Accuracy:** ~80%

---

## 🚀 How to Run

### 1. Clone Repo
```bash
git clone https://github.com/Deepak-singh3410/major-project-vocallab.git
cd vocal

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Generate Spectrograms
python generate_spectrograms.py

### 4 Run Inference
python hierarchical_inference.py

Train Models
python train_model.py
python train_multiclass_cnn.py
python train_parkinsons_model.
