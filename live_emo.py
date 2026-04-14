import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
import tkinter.messagebox as messagebox

# ==========================
# LOAD MODEL
# ==========================
print("Loading model...")
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

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==========================
# CHOOSE DETECTION MODE
# ==========================
root = Tk()
root.withdraw()  # Hide the root window

choice = messagebox.askyesno(
    "Emotion Detection Mode",
    "Yes = Live Camera Detection\nNo = Static Image Detection"
)

if choice:
    # ===== LIVE CAMERA MODE =====
    root.destroy()
    print("Starting live camera detection... Press 'q' to quit.")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam!")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face.astype('float32') / 255.0
            face = np.reshape(face, (1, IMG_SIZE, IMG_SIZE, 1))

            prediction = model.predict(face, verbose=0)
            emotion_idx = np.argmax(prediction)
            emotion = emotion_labels[emotion_idx]
            confidence = prediction[0][emotion_idx] * 100

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,0), 2)

        cv2.imshow("Emotion Detection - Live Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    # ===== STATIC IMAGE MODE =====
    image_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
    )
    root.destroy()
    
    # If no image selected, exit
    if not image_path or not os.path.exists(image_path):
        print("No image selected. Exiting.")
        exit()

    # Load the image (first try as color, then as grayscale if needed)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        exit()

    # Load as grayscale for processing
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if gray is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    print(f"Image shape: {gray.shape}")
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No faces detected. Using entire image as face...")
        h, w = gray.shape
        faces = [(0, 0, w, h)]
    else:
        print(f"Detected {len(faces)} face(s) in the image.")

    # Create color version for display (convert grayscale back to BGR if needed)
    if len(frame.shape) == 2:
        display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        display_frame = frame.copy()

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        
        # Debug: show original range
        print(f"Face pixel range: min={face.min()}, max={face.max()}")
        
        face = face.astype('float32') / 255.0
        face = np.reshape(face, (1, IMG_SIZE, IMG_SIZE, 1))

        prediction = model.predict(face, verbose=0)
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        confidence = prediction[0][emotion_idx] * 100

        print(f"Predicted emotion: {emotion} (confidence: {confidence:.2f}%)")
        print(f"All predictions: {dict(zip(emotion_labels, prediction[0]))}")

        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(display_frame, f"{emotion} ({confidence:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,0), 2)

    # Display the result
    cv2.imshow("Emotion Detection - Image", display_frame)

    print("Press any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
