import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==============================
# LOAD NORMALIZED DATASET
# ==============================

DATASET_PATH = "dataset/static/processed/static_landmarks_normalized.csv"

# low_memory=False removes dtype warning
df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded successfully!")
print(f"Total samples: {len(df)}")

# ==============================
# FEATURES AND LABELS
# ==============================

# All landmark columns
X = df.iloc[:, :-1]

# Convert ALL labels to string
# This fixes mixed int/string label issue
y = df.iloc[:, -1].astype(str)

print("\nUnique Labels:")
print(sorted(y.unique()))

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))

# ==============================
# CREATE RANDOM FOREST MODEL
# ==============================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# ==============================
# TRAIN MODEL
# ==============================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")

# ==============================
# PREDICTIONS
# ==============================

print("\nMaking predictions...")

y_pred = model.predict(X_test)

# ==============================
# EVALUATION
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== RESULTS ==========")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# SAVE MODEL
# ==============================

MODEL_PATH = "static_landmark_model.pkl"

joblib.dump(model, MODEL_PATH)

print("\n========== MODEL SAVED ==========")
print(f"Model saved successfully at: {MODEL_PATH}")