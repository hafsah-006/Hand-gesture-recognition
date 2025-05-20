import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2
import os
import numpy as np
from sklearn.utils import shuffle

loadedImages = []
outputVectors = []

# Paths and gesture folder names
base_path = r'D:\Hafsah\cnn dataset'
gesture_folders = ['up', 'down', 'left', 'right', 'flip']

# Number of images per folder
num_images_per_folder = 101

for idx, folder in enumerate(gesture_folders):
    folder_path = os.path.join(base_path, folder)

    for i in range(num_images_per_folder):
        filename = f'fist_{i}.png'  # Your images are named "fist_#.png"
        full_path = os.path.join(folder_path, filename)

        image = cv2.imread(full_path)
        if image is None:
            print(f"[Warning] Could not read image: {full_path}")
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (100, 89))  # Resize to (width, height)
        loadedImages.append(resized_image.reshape(89, 100, 1))

        # Create one-hot vector for this image
        one_hot_vector = [0] * len(gesture_folders)
        one_hot_vector[idx] = 1
        outputVectors.append(one_hot_vector)

print(f"✅ Total images loaded: {len(loadedImages)}")

testImages = []
testLabels = []

for idx, folder in enumerate(gesture_folders):
    folder_path = os.path.join(base_path, folder + 'Test')  # Assuming test folders are like 'upTest', 'downTest', etc.

    for i in range(num_images_per_folder):
        filename = f'fist_{i}.png'  # Assuming naming pattern same as training ('fist_#.png')
        full_path = os.path.join(folder_path, filename)

        image = cv2.imread(full_path)
        if image is None:
            print(f"[Warning] Could not read image: {full_path}")
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (100, 89))  # Resize to (width, height)
        testImages.append(resized_image.reshape(89, 100, 1))

        # Create one-hot vector label for the gesture
        one_hot_vector = [0] * len(gesture_folders)
        one_hot_vector[idx] = 1
        testLabels.append(one_hot_vector)

print(f"✅ Total test images loaded: {len(testImages)}")

# Convert lists to numpy arrays
X_train = np.array(loadedImages, dtype=np.float32) / 255.0  # Normalize pixel values
y_train = np.array(outputVectors)
X_test = np.array(testImages, dtype=np.float32) / 255.0  # Normalize pixel values
y_test = np.array(testLabels)

# Reduce the number of pooling layers to prevent negative output size
model = Sequential([
    Conv2D(32, (2, 2), activation='relu', input_shape=(89, 100, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (2, 2), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (2, 2), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (2, 2), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    # Removed some max pooling layers to prevent negative dimensions
    Conv2D(128, (2, 2), activation='relu'),
    Flatten(),
    Dense(1000, activation='relu'),
    Dropout(0.5),  # Reduced dropout rate
    Dense(5, activation='softmax')  # For 5 gesture classes
])

# Print model summary
model.summary()

# Model compilation
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model training
history = model.fit(
    X_train, y_train,  # Use the numpy arrays we created
    epochs=50,
    validation_data=(X_test, y_test),
    batch_size=32,
    verbose=1
)

# Save model with correct extension
model.save("TrainedModel/GestureRecogModel.h5")  # Use .h5 format for better compatibility
