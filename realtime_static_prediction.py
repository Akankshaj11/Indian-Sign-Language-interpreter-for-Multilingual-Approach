


import cv2
import mediapipe as mp
import numpy as np
import joblib


class StaticPredictor:

    def __init__(self):

        # ==============================
        # LOAD MODEL
        # ==============================

        self.model = joblib.load("static_landmark_model.pkl")

        print("Static Model Loaded Successfully!")

        # ==============================
        # MEDIAPIPE
        # ==============================

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    # ============================================
    # Predict Static Gesture
    # ============================================

    def predict(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(rgb)

        prediction = "No Hand Detected"

        confidence = 0

        if results.multi_hand_landmarks:

            row = []

            for hand_landmarks in results.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                for lm in hand_landmarks.landmark:

                    row.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

            # Pad if one hand

            while len(row) < 126:
                row.extend([0,0,0])

            row = row[:126]

            # ==========================
            # NORMALIZATION
            # ==========================

            landmarks = np.array(row).reshape(42,3)

            wrist = landmarks[0]

            landmarks = landmarks - wrist

            max_value = np.max(np.abs(landmarks))

            if max_value != 0:

                landmarks = landmarks / max_value

            normalized = landmarks.flatten().reshape(1,-1)

            # ==========================
            # PREDICT
            # ==========================

            prediction = self.model.predict(normalized)[0]

            # Optional confidence

            if hasattr(self.model, "predict_proba"):

                confidence = (
                    np.max(
                        self.model.predict_proba(normalized)
                    ) * 100
                )

            else:

                confidence = 100

        return prediction, confidence, frame

    # ============================================
    # Cleanup
    # ============================================

    def close(self):

        self.hands.close()