import cv2
import mediapipe as mp
import numpy as np
import pickle

with open('gesture_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    SIGNS = data['signs']

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

print("Camera chal raha hai! Q dabao band karne ke liye.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hl in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hl.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            features = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(features)[0]
            confidence = model.predict_proba(features)[0][prediction] * 100
            sign_name = SIGNS[prediction]

            cv2.putText(frame, f"Sign: {sign_name.upper()}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Haath dikhao!", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Gesture Genius - Live Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
