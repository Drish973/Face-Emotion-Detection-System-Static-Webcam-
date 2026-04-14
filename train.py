import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================
# CONFIG
# ==========================
DATASET_PATH = "input/CK+48"
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 30  # Reasonable number
MODEL_SAVE_PATH = "emotion_model.h5"

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset path not found: {DATASET_PATH}.\n"
        "Please place your emotion folders (anger, contempt, etc.) under this path."
    )

# ==========================
# DATA GENERATOR
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

num_classes = train_data.num_classes
print("Detected classes:", train_data.class_indices)
print(f"Training on {train_data.samples} images, validating on {val_data.samples} images.")

# ==========================
# MODEL ARCHITECTURE
# ==========================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')  # 7 emotions
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# TRAIN
# ==========================
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Calculate steps properly
steps_per_epoch = train_data.samples // BATCH_SIZE
validation_steps = val_data.samples // BATCH_SIZE

# Ensure at least 1 step
steps_per_epoch = max(1, steps_per_epoch)
validation_steps = max(1, validation_steps)

print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr]
)

model.save(MODEL_SAVE_PATH)
print("Model saved successfully.")

# Print final accuracy
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(".2f")
print(".2f")
