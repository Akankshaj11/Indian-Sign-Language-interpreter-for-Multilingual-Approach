import cv2
import os

LETTER = "Z"   # change A → Z whenever recording a new letter

SAVE_DIR = f"dataset/static/{LETTER}"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press 's' to save image")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)

    # Display letter on screen
    cv2.putText(frame, f"Letter: {LETTER}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Static Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.imwrite(f"{SAVE_DIR}/{count}.jpg", frame)
        print(f"Saved image {count}")
        count += 1
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
