import pandas as pd
import numpy as np

# Input and output paths
INPUT_CSV = "dataset/static/processed/static_landmarks.csv"
OUTPUT_CSV = "dataset/static/processed/static_landmarks_normalized.csv"

# Load dataset
df = pd.read_csv(INPUT_CSV)

# Separate features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

normalized_data = []

for row in X:

    row = row.astype(float)

    # Convert to numpy array
    landmarks = np.array(row).reshape(42, 3)

    # Wrist landmark (first landmark)
    wrist = landmarks[0]

    # Shift all landmarks relative to wrist
    landmarks = landmarks - wrist

    # Scale normalization
    max_value = np.max(np.abs(landmarks))

    if max_value != 0:
        landmarks = landmarks / max_value

    # Flatten back
    normalized_row = landmarks.flatten().tolist()

    normalized_data.append(normalized_row)

# Create DataFrame
normalized_df = pd.DataFrame(normalized_data)

# Add labels
normalized_df["label"] = y

# Save normalized dataset
normalized_df.to_csv(OUTPUT_CSV, index=False)

print("Normalized dataset created successfully!")
print(f"Saved at: {OUTPUT_CSV}")
print(f"Total samples: {len(normalized_df)}")