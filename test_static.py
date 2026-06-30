# import cv2

# from realtime_static_prediction import StaticPredictor

# predictor = StaticPredictor()

# cap = cv2.VideoCapture(0)

# while True:

#     ret, frame = cap.read()

#     frame = cv2.flip(frame,1)

#     prediction, confidence, frame = predictor.predict(frame)

#     cv2.putText(
#         frame,
#         f"{prediction} ({confidence:.1f}%)",
#         (20,40),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0,255,0),
#         2
#     )

#     cv2.imshow("Static",frame)

#     if cv2.waitKey(1)==27:
#         break

# cap.release()

# predictor.close()

# cv2.destroyAllWindows()











import cv2

from realtime_static_prediction import StaticPredictor

predictor = StaticPredictor()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    prediction, confidence, frame = predictor.predict(frame)

    cv2.putText(
        frame,
        f"{prediction}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"{confidence:.1f}%",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,0),
        2
    )

    cv2.imshow("Static Prediction",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()

predictor.close()

cv2.destroyAllWindows()