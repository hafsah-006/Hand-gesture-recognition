import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model with verification
try:
    model = load_model("TrainedModel/GestureRecogModel.h5")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    exit()

gesture_labels = ['up', 'down', 'left', 'right', 'flip']

# Initialize camera with explicit settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Camera access denied - check permissions or try different index")
    exit()

# Adjusted ROI coordinates (relative to 640x480 frame)
roi_top, roi_bottom = 100, 300  # 200px tall
roi_left, roi_right = 220, 420  # 200px wide


def preprocess_frame(roi):
    """Match training preprocessing exactly"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 89))  # Width=100, Height=89
    return resized.reshape(1, 89, 100, 1).astype('float32') / 255.0


print("Starting gesture recognition... Press 'Q' to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame capture error")
            break

        # Mirror and extract ROI
        frame = cv2.flip(frame, 1)
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]

        if roi.size == 0:
            print("⚠️ Invalid ROI coordinates")
            break

        # Process and predict
        processed = preprocess_frame(roi)
        predictions = model.predict(processed, verbose=0)[0]
        class_id = np.argmax(predictions)
        confidence = np.max(predictions)

        # Display results
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{gesture_labels[class_id]} ({confidence * 100:.1f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show ROI preview
        cv2.imshow("ROI Preview", cv2.resize(roi, (200, 200)))
        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) in [ord('q'), 27]:  # 27=ESC
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
