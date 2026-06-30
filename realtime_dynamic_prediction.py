

import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
from tensorflow.keras.models import load_model


class DynamicPredictor:

    def __init__(self):

        # =====================================
        # LOAD MODEL
        # =====================================

        self.model = load_model("dynamic_model.keras")

        self.encoder = joblib.load(
            "dynamic_label_encoder.pkl"
        )

        print("Dynamic Model Loaded Successfully!")

        # =====================================
        # MEDIAPIPE
        # =====================================

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # =====================================
        # SETTINGS
        # =====================================

        self.SEQUENCE_LENGTH = 30

        self.CONFIDENCE_THRESHOLD = 0.70

        # =====================================
        # BUFFERS
        # =====================================

        self.sequence = deque(maxlen=30)

        self.prediction_history = deque(maxlen=10)

        self.current_prediction = ""

        self.current_confidence = 0

        # =====================================
        # MISSED FRAME LOGIC
        # =====================================

        self.consecutive_missed_frames = 0

        self.MAX_MISSED_FRAMES = 15

    # =====================================
    # NORMALIZE FRAME
    # =====================================

    def normalize_frame(self, frame_data):

        landmarks = np.array(
            frame_data
        ).reshape(21,3)

        wrist = landmarks[0]

        landmarks = landmarks - wrist

        max_value = np.max(np.abs(landmarks))

        if max_value != 0:

            landmarks = landmarks / max_value

        return landmarks.flatten()

    # =====================================
    # PREDICT
    # =====================================

    def predict(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(rgb)

        prediction = "Show Gesture"

        confidence = 0

        # =====================================
        # HAND FOUND
        # =====================================

        if results.multi_hand_landmarks:

            self.consecutive_missed_frames = 0

            hand = results.multi_hand_landmarks[0]

            self.mp_draw.draw_landmarks(
                frame,
                hand,
                self.mp_hands.HAND_CONNECTIONS
            )

            row = []

            for lm in hand.landmark:

                row.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            row = self.normalize_frame(row)

            self.sequence.append(row)

            if len(self.sequence) == self.SEQUENCE_LENGTH:

                sample = np.expand_dims(
                    np.array(self.sequence),
                    axis=0
                )

                probs = self.model.predict(
                    sample,
                    verbose=0
                )[0]

                idx = np.argmax(probs)

                confidence = probs[idx]

                prediction = self.encoder.inverse_transform(
                    [idx]
                )[0]                # =====================================
                # CONFIDENCE CHECK
                # =====================================

                if confidence >= self.CONFIDENCE_THRESHOLD:

                    self.prediction_history.append(prediction)

                    # Majority Voting
                    values, counts = np.unique(
                        self.prediction_history,
                        return_counts=True
                    )

                    prediction = values[np.argmax(counts)]

                    self.current_prediction = prediction
                    self.current_confidence = confidence * 100

                else:

                    # Low confidence:
                    # Keep previous prediction instead of flickering
                    prediction = self.current_prediction
                    confidence = self.current_confidence / 100

            else:

                # Need more frames
                prediction = "Show Gesture"

        # =====================================
        # NO HAND DETECTED
        # =====================================

        else:

            self.consecutive_missed_frames += 1

            # Keep previous prediction for a short time
            if self.consecutive_missed_frames <= self.MAX_MISSED_FRAMES:

                prediction = self.current_prediction
                confidence = self.current_confidence / 100

            else:

                # Reset after grace period
                self.sequence.clear()
                self.prediction_history.clear()

                self.current_prediction = ""
                self.current_confidence = 0

                prediction = "Show Gesture"
                confidence = 0

        # =====================================
        # DISPLAY CONFIDENCE
        # =====================================

        confidence = confidence * 100 if confidence <= 1 else confidence

        return prediction, confidence, frame

    # =====================================
    # RESET
    # =====================================

    def reset(self):

        self.sequence.clear()

        self.prediction_history.clear()

        self.current_prediction = ""

        self.current_confidence = 0

        self.consecutive_missed_frames = 0

    # =====================================
    # CLOSE
    # =====================================

    def close(self):

        self.hands.close()
