import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands

# MediaPipe Hands Configuration
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.2
)

# Paths
DATASET_PATH = "dataset/static/raw"
OUTPUT_CSV = "dataset/static/processed/static_landmarks.csv"

# Storage
data = []

total_images = 0
detected_images = 0
failed_images = 0

# Read labels/classes
labels = os.listdir(DATASET_PATH)

for label in labels:

    label_path = os.path.join(DATASET_PATH, label)

    # Skip non-folder files
    if not os.path.isdir(label_path):
        continue

    print(f"\nProcessing label: {label}")

    # Read all images in label folder
    for image_name in os.listdir(label_path):

        total_images += 1

        image_path = os.path.join(label_path, image_name)

        # Load image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load image: {image_path}")
            failed_images += 1
            continue

        # Resize image for better MediaPipe detection
        image = cv2.resize(image, (640, 480))

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe
        results = hands.process(image_rgb)

        row = []

        # If hand landmarks detected
        if results.multi_hand_landmarks:

            print(f"Hand detected in: {image_name}")

            detected_images += 1

            for hand_landmarks in results.multi_hand_landmarks:

                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

            # Padding if only one hand detected
            while len(row) < 126:
                row.extend([0, 0, 0])

            # Add label at end
            row.append(label)

            # Save row
            data.append(row)

        else:
            print(f"No hand detected: {image_path}")

# Final Statistics
print("\n========== FINAL REPORT ==========")
print(f"Total images processed : {total_images}")
print(f"Successful detections  : {detected_images}")
print(f"Failed detections      : {total_images - detected_images}")
print(f"Failed image loads     : {failed_images}")

# Save CSV
with open(OUTPUT_CSV, mode='w', newline='') as file:

    writer = csv.writer(file)

    # Create header
    header = []

    for i in range(126):
        header.append(f"value_{i}")

    header.append("label")

    writer.writerow(header)

    # Write data rows
    writer.writerows(data)

print("\nCSV file created successfully!")
print(f"Saved at: {OUTPUT_CSV}")