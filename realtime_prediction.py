import cv2
import mediapipe as mp
import numpy as np
import joblib

# ==============================
# LOAD TRAINED MODEL
# ==============================

MODEL_PATH = "static_landmark_model.pkl"

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")

# ==============================
# MEDIAPIPE SETUP
# ==============================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==============================
# START WEBCAM
# ==============================

cap = cv2.VideoCapture(0)

print("Starting webcam...")

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read webcam frame")
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = hands.process(rgb_frame)

    prediction_text = "No Hand Detected"

    if results.multi_hand_landmarks:

        row = []

        # Draw landmarks
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract landmarks
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

        # Padding if only one hand
        while len(row) < 126:
            row.extend([0, 0, 0])

        # Trim extra values if needed
        row = row[:126]

        # ==============================
        # NORMALIZATION
        # ==============================

        landmarks = np.array(row).reshape(42, 3)

        # Wrist landmark
        wrist = landmarks[0]

        # Shift relative to wrist
        landmarks = landmarks - wrist

        # Scale normalization
        max_value = np.max(np.abs(landmarks))

        if max_value != 0:
            landmarks = landmarks / max_value

        # Flatten again
        normalized_row = landmarks.flatten().reshape(1, -1)

        # ==============================
        # PREDICTION
        # ==============================

        prediction = model.predict(normalized_row)[0]

        prediction_text = f"Prediction: {prediction}"

    # ==============================
    # DISPLAY TEXT
    # ==============================

    cv2.putText(
        frame,
        prediction_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show webcam
    cv2.imshow("Real-Time Sign Prediction", frame)

    # Exit on ESC key
    key = cv2.waitKey(1)

    if key == 27:
        break

# ==============================
# CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()