import cv2
import numpy as np
import pickle

with open('gesture_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    SIGNS = data['signs']

cap = cv2.VideoCapture(0)
IMG_SIZE = 64

print("Camera chal raha hai! Q dabao band karne ke liye.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = w//4, h//4, 3*w//4, 3*h//4

    hand_roi = frame[y1:y2, x1:x2]
    resized = cv2.resize(hand_roi, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = gray / 255.0
    features = normalized.flatten().reshape(1, -1)

    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][prediction] * 100
    sign_name = SIGNS[prediction]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Sign: {sign_name.upper()}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Haath green box mein rakhein", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.imshow('Gesture Genius - Live Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()