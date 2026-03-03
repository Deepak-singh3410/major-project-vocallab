import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Input,
    BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/spectrograms_multiclass"
IMG_SIZE = (224, 224)   # 🔥 Reduced for better generalization
BATCH_SIZE = 16
EPOCHS = 50

# -----------------------------
# LOAD IMAGES INTO MEMORY
# -----------------------------
images = []
labels = []

class_names = sorted(os.listdir(DATA_PATH))
class_indices = {name: idx for idx, name in enumerate(class_names)}

print("Class mapping:", class_indices)

for class_name in class_names:
    folder = os.path.join(DATA_PATH, class_name)

    for file in os.listdir(folder):
        if file.endswith(".png"):
            img_path = os.path.join(folder, file)

            img = load_img(img_path, target_size=IMG_SIZE)
            img = img_to_array(img)

            images.append(img)
            labels.append(class_indices[class_name])

images = np.array(images)
labels = np.array(labels)

# Normalize
images = images / 255.0

print("Total samples:", len(images))
print("Class distribution:", np.bincount(labels))

# -----------------------------
# STRATIFIED SPLIT
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    images,
    labels,
    test_size=0.2,
    stratify=labels,
    shuffle=True,
    random_state=42
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Train distribution:", np.bincount(y_train))
print("Val distribution:", np.bincount(y_val))

# -----------------------------
# CLASS WEIGHTS (🔥 Important)
# -----------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# -----------------------------
# MODEL ARCHITECTURE
# -----------------------------
model = Sequential([
    Input((224, 224, 3)),

    Conv2D(32, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    Conv2D(64, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    Conv2D(128, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# TRAINING
# -----------------------------
early_stop = EarlyStopping(
    patience=8,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    class_weight=class_weights   # 🔥 Added
)

# -----------------------------
# FINAL EVALUATION
# -----------------------------
val_loss, val_acc = model.evaluate(X_val, y_val)

print("\nFinal Validation Accuracy:", val_acc)
print("Final Validation Loss:", val_loss)

# Confusion Matrix + Report
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred_classes))

print("\nClassification Report:")
print(classification_report(y_val, y_pred_classes))

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("data/multiclass_cnn.keras")
print("Multiclass CNN saved successfully.")