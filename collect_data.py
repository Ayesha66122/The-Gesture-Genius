import cv2
import mediapipe as mp
import numpy as np
import os

SIGNS = ['hello', 'thankyou', 'yes', 'no', 'one', 'two', 'three', 'four', 'five']
SAMPLES_PER_SIGN = 100

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

os.makedirs('dataset', exist_ok=True)
cap = cv2.VideoCapture(0)

for sign in SIGNS:
    print(f"\n>>> '{sign}' ke liye tayar ho jao!")
    print("SPACE dabao shuru karne ke liye")

    data = []
    collecting = False

    while len(data) < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

                if collecting:
                    landmarks = []
                    for lm in hl.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    data.append(landmarks)

        color = (0, 255, 0) if collecting else (0, 165, 255)
        status = f"Collecting: {len(data)}/{SAMPLES_PER_SIGN}" if collecting else "SPACE dabao"
        cv2.putText(frame, f"Sign: {sign.upper()}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, status, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow('Gesture Genius - Data Collector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            collecting = not collecting
        elif key == ord('q'):
            break

    np.save(f'dataset/{sign}.npy', np.array(data))
    print(f"✓ {sign} done! {len(data)} samples save ho gaye")

cap.release()
cv2.destroyAllWindows()
print("\nSab signs collect ho gaye!")
