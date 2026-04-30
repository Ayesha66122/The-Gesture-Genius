import cv2
import numpy as np
import os

SIGNS = ['hello', 'thankyou', 'yes', 'no', 'one', 'two', 'three', 'four', 'five']
SAMPLES_PER_SIGN = 100

os.makedirs('dataset', exist_ok=True)
cap = cv2.VideoCapture(0)

IMG_SIZE = 64

for sign in SIGNS:
    print(f"\n>>> '{sign}' ke liye tayar ho jao!")
    print("SPACE dabao shuru karne ke liye, Q se quit")

    data = []
    collecting = False

    while len(data) < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Haath ka area — screen ke beech ka hissa
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = w//4, h//4, 3*w//4, 3*h//4

        # Haath ka box
        hand_roi = frame[y1:y2, x1:x2]
        resized = cv2.resize(hand_roi, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        normalized = gray / 255.0

        if collecting:
            data.append(normalized.flatten())

        # Screen pe box dikhao
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        color = (0, 255, 0) if collecting else (0, 165, 255)
        status = f"Collecting: {len(data)}/{SAMPLES_PER_SIGN}" if collecting else "SPACE dabao - haath box mein rakhein"
        cv2.putText(frame, f"Sign: {sign.upper()}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, status, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow('Gesture Genius - Data Collector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            collecting = not collecting
        elif key == ord('q'):
            break

    if len(data) > 0:
        np.save(f'dataset/{sign}.npy', np.array(data))
        print(f"✓ {sign} done! {len(data)} samples save ho gaye")

cap.release()
cv2.destroyAllWindows()
print("\nSab signs collect ho gaye! dataset/ folder check karo.")