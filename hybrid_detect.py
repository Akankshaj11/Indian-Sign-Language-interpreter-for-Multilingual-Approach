import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load models
dynamic_model = load_model("gesture_lstm.h5")
static_model = load_model("static_letters.h5")

DYNAMIC_LABELS = ['bye', 'hello', 'namaste', 'thank_you', 'yes']
STATIC_LABELS = sorted(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

mode = "dynamic"
sequence = []

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        if mode == "dynamic":
            data = []
            for lm in hand.landmark:
                data.extend([lm.x, lm.y, lm.z])
            sequence.append(data)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                pred = dynamic_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                word = DYNAMIC_LABELS[np.argmax(pred)]
                cv2.putText(frame, f"WORD: {word}", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        else:  # static
            img = cv2.resize(frame, (64,64))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            pred = static_model.predict(img, verbose=0)[0]
            letter = STATIC_LABELS[np.argmax(pred)]
            cv2.putText(frame, f"LETTER: {letter}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.putText(frame, f"Mode: {mode.upper()} (D/S)",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Hybrid Sign Language System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        mode = "dynamic"
        sequence = []
    elif key == ord('s'):
        mode = "static"
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()