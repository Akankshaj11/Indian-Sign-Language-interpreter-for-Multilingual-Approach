import cv2
import mediapipe as mp
import numpy as np
import os

# CHANGE THIS for each gesture when collecting
GESTURE = "yes"

SAVE_DIR = f"dataset/dynamic/{GESTURE}"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

SEQUENCE_LENGTH = 30   # 30 frame sequences
sequence = []
sample_no = 0

cap = cv2.VideoCapture(0)
print("Press 'r' to start recording a 30-frame sequence.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        frame_data = []
        for p in lm.landmark:
            frame_data.extend([p.x, p.y, p.z])

        if len(sequence) < SEQUENCE_LENGTH:
            sequence.append(frame_data)

        # Save when 30 frames (1 sequence) completed
        if len(sequence) == SEQUENCE_LENGTH:
            np.save(f"{SAVE_DIR}/{sample_no}.npy", np.array(sequence))
            print(f"Saved sample: {sample_no}")
            sample_no += 1
            sequence = []

    cv2.imshow("Dynamic Dataset Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('r'):   # restart sequence capture
        sequence = []

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
