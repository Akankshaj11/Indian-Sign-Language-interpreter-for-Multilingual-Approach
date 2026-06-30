


import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ==========================================================
# SETTINGS
# ==========================================================
GESTURE = "tired"
SEQUENCE_LENGTH = 30
COOLDOWN_TIME = 2.0  # Time in seconds to wait between recordings

SAVE_DIR = os.path.join("dataset", "dynamic", "raw", GESTURE)
os.makedirs(SAVE_DIR, exist_ok=True)

# Find next sample number
existing = [
    int(f.split(".")[0])
    for f in os.listdir(SAVE_DIR)
    if f.endswith(".npy") and f.split(".")[0].isdigit()
]
sample_no = max(existing) + 1 if existing else 0

# MediaPipe Setup (Updated for 2 hands)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.6, 
    min_tracking_confidence=0.5
)

sequence = []
recording = False
status = "READY"
last_record_time = 0

def normalize_frame(frame_data):
    # Reshape for 21 landmarks x 3 coordinates
    landmarks = np.array(frame_data).reshape(21, 3)
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    max_val = np.max(np.abs(landmarks))
    return (landmarks / max_val if max_val != 0 else landmarks).flatten()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_time = time.time()

    # Capture data if hands are present
    if results.multi_hand_landmarks:
        # Determine if we are recording the primary hand (index 0)
        hand = results.multi_hand_landmarks[0]
        
        # Draw all detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if status == "READY" and (current_time - last_record_time) > COOLDOWN_TIME:
            recording = True
            status = "RECORDING"
            sequence = []

        if recording:
            row = []
            for lm in hand.landmark:
                row.extend([lm.x, lm.y, lm.z])
            sequence.append(normalize_frame(row))

            progress = int((len(sequence)/SEQUENCE_LENGTH)*300)
            cv2.rectangle(frame, (170, 440), (170 + progress, 460), (0, 255, 0), -1)

            if len(sequence) == SEQUENCE_LENGTH:
                np.save(os.path.join(SAVE_DIR, f"{sample_no}.npy"), np.array(sequence))
                print(f"Saved Sample {sample_no} (Hands detected: {len(results.multi_hand_landmarks)})")
                sample_no += 1
                recording = False
                status = "READY"
                last_record_time = time.time()

    # UI Overlay
    cv2.putText(frame, f"Gesture: {GESTURE} | Samples: {sample_no}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Status: {status}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if recording else (255, 255, 255), 2)
    
    cv2.imshow("Dynamic Dataset Collector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()