

# import cv2
# import mediapipe as mp
# import os

# LETTER = "A"
# SAVE_DIR = f"dataset/static/{LETTER}"
# os.makedirs(SAVE_DIR, exist_ok=True)

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=2,
#     min_detection_confidence=0.7
# )

# cap = cv2.VideoCapture(0)

# count = 0
# IMG_SIZE = 224   # perfect for CNN models

# print("Auto hand capture started...")
# print("Press 'q' to quit")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(rgb)

#     if results.multi_hand_landmarks:

#         for hand_landmarks in results.multi_hand_landmarks:

#             h, w, _ = frame.shape

#             # Get bounding box
#             x_coords = [lm.x for lm in hand_landmarks.landmark]
#             y_coords = [lm.y for lm in hand_landmarks.landmark]

#             xmin = int(min(x_coords) * w)
#             xmax = int(max(x_coords) * w)
#             ymin = int(min(y_coords) * h)
#             ymax = int(max(y_coords) * h)

#             # Add margin
#             margin = 40
#             xmin = max(0, xmin - margin)
#             ymin = max(0, ymin - margin)
#             xmax = min(w, xmax + margin)
#             ymax = min(h, ymax + margin)

#             hand_crop = frame[ymin:ymax, xmin:xmax]

#             # Avoid empty crop
#             if hand_crop.size == 0:
#                 continue

#             # Resize for uniform dataset
#             hand_crop = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))

#             # Save image
#             cv2.imwrite(f"{SAVE_DIR}/{count}.jpg", hand_crop)
#             count += 1

#             # Draw rectangle (for visualization only)
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

#     cv2.putText(frame, f"Saved: {count}", (20,40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#     cv2.imshow("Pro Collector", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()








import cv2
import mediapipe as mp
import os

LETTER = "Z"   # Change per letter

SAVE_DIR = f"dataset/static/letters/{LETTER}"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75
)

cap = cv2.VideoCapture(0)

IMG_SIZE = 224
MARGIN = 40

count = len(os.listdir(SAVE_DIR))

print("Intelligent Collector Started...")
print("Supports ONE and TWO hands")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        h, w, _ = frame.shape

        xmin_global, ymin_global = w, h
        xmax_global, ymax_global = 0, 0

        # 🔥 combine bounding boxes of ALL hands
        for hand_landmarks in results.multi_hand_landmarks:

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            xmin = int(min(x_coords) * w)
            xmax = int(max(x_coords) * w)
            ymin = int(min(y_coords) * h)
            ymax = int(max(y_coords) * h)

            xmin_global = min(xmin_global, xmin)
            ymin_global = min(ymin_global, ymin)
            xmax_global = max(xmax_global, xmax)
            ymax_global = max(ymax_global, ymax)

        # Add margin
        xmin_global = max(0, xmin_global - MARGIN)
        ymin_global = max(0, ymin_global - MARGIN)
        xmax_global = min(w, xmax_global + MARGIN)
        ymax_global = min(h, ymax_global + MARGIN)

        crop = frame[ymin_global:ymax_global, xmin_global:xmax_global]

        if crop.size != 0:

            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

            cv2.imwrite(f"{SAVE_DIR}/{count}.jpg", crop)
            count += 1

            # draw rectangle for visualization
            cv2.rectangle(frame,
                          (xmin_global, ymin_global),
                          (xmax_global, ymax_global),
                          (0,255,0), 2)

    cv2.putText(frame, f"Saved: {count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Intelligent Collector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
