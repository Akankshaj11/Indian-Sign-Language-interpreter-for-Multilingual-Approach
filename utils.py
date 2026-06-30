import cv2
import numpy as np
import time


# ==========================================
# FPS
# ==========================================

class FPSCounter:

    def __init__(self):
        self.prev_time = time.time()

    def get_fps(self):

        current = time.time()

        fps = 1 / (current - self.prev_time)

        self.prev_time = current

        return int(fps)


# ==========================================
# STATIC NORMALIZATION
# 42 landmarks (2 hands)
# ==========================================

def normalize_static(row):

    row = row[:126]

    while len(row) < 126:
        row.extend([0, 0, 0])

    landmarks = np.array(row).reshape(42, 3)

    wrist = landmarks[0]

    landmarks = landmarks - wrist

    max_val = np.max(np.abs(landmarks))

    if max_val != 0:
        landmarks = landmarks / max_val

    return landmarks.flatten().reshape(1, -1)


# ==========================================
# DYNAMIC NORMALIZATION
# 21 landmarks (1 hand)
# ==========================================

def normalize_dynamic(frame_data):

    landmarks = np.array(frame_data).reshape(21, 3)

    wrist = landmarks[0]

    landmarks = landmarks - wrist

    max_val = np.max(np.abs(landmarks))

    if max_val != 0:
        landmarks = landmarks / max_val

    return landmarks.flatten()


# ==========================================
# CLAHE Lighting Enhancement
# ==========================================

def enhance_lighting(frame):

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    frame = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    return frame


# ==========================================
# Draw Transparent Rectangle
# ==========================================

def draw_overlay(frame, pt1, pt2, alpha=0.45):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        pt1,
        pt2,
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )


# ==========================================
# Confidence Color
# ==========================================

def confidence_color(conf):

    if conf >= 90:
        return (0,255,0)

    if conf >= 75:
        return (0,255,255)

    return (0,0,255)