import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("C:\\Users\\ds469\\OneDrive\\Pictures\\Desktop\\Majour_project\\vocal\\data\\parkinsions\\parkinsons.csv")

print("Dataset shape:", data.shape)
print(data.head())

# -----------------------------
# CLEAN DATA
# -----------------------------
# Drop name column
if "name" in data.columns:
    data = data.drop(columns=["name"])

# Target
y = data["status"]

# Features
X = data.drop(columns=["status"])

print("\nFeature Columns:", list(X.columns))

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
os.makedirs("data/parkinsons", exist_ok=True)

joblib.dump(model, "data/parkinsons/parkinsons_model.pkl")
print("\nParkinson's model saved successfully.")