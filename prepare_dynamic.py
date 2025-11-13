import numpy as np
import os
from sklearn.model_selection import train_test_split

DATA_PATH = "dataset/dynamic"

gestures = os.listdir(DATA_PATH)
gesture_to_label = {g:i for i, g in enumerate(gestures)}

X = []
y = []

for gesture in gestures:
    folder = os.path.join(DATA_PATH, gesture)
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            seq = np.load(os.path.join(folder, file))
            X.append(seq)
            y.append(gesture_to_label[gesture])

X = np.array(X)
y = np.array(y)

print("Dataset loaded:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", gesture_to_label)

# Split (small training set for now)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Save for training step
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Saved X_train, X_test, y_train, y_test")
