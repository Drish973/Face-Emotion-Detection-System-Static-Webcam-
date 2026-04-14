import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==========================
# LOAD MODEL
# ==========================
model = load_model("emotion_model.h5")

emotion_labels = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "sadness",
    "surprise"
]

IMG_SIZE = 48

# ==========================
# STATIC DETECTION
# ==========================
DATASET_PATH = "input/CK+48"

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

results = []

for emotion_folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, emotion_folder)
    if not os.path.isdir(folder_path):
        continue

    print(f"Processing {emotion_folder}...")

    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize directly (assuming images are already face-cropped)
        face = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = np.reshape(face, (1, IMG_SIZE, IMG_SIZE, 1))

        prediction = model.predict(face, verbose=0)
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        results.append({
            'image': image_path,
            'true_emotion': emotion_folder,
            'predicted_emotion': predicted_emotion,
            'confidence': float(confidence)
        })

        print(f"Image: {image_path}")
        print(f"True: {emotion_folder}, Predicted: {predicted_emotion}, Confidence: {confidence:.2f}")
        print("-" * 50)

# Save results to file
import json
with open('detection_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Detection complete. Results saved to detection_results.json")