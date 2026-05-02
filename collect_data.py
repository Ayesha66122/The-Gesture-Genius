import cv2
import numpy as np
import os
import time

SIGNS = ['hello', 'thankyou', 'yes', 'no', 'one', 'two', 'three', 'four', 'five']
SAMPLES_PER_SIGN = 100

os.makedirs('dataset', exist_ok=True)
cap = cv2.VideoCapture(0)

IMG_SIZE = 64

for sign in SIGNS:
    print(f"\n>>> '{sign}' ke liye tayar ho jao!")
    print("ENTER dabao — camera window khulegi")
    input()

    data = []

    while len(data) < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        x1, y1 = w//4, h//4
        x2, y2 = 3*w//4, 3*h//4

        # Green box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Text
        cv2.putText(frame, f"Sign: {sign.upper()}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Samples: {len(data)}/{SAMPLES_PER_SIGN}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, "SPACE=Start  Q=Quit", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

        cv2.imshow('Gesture Genius - Data Collector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # Collecting
            hand_roi = frame[y1:y2, x1:x2]
            resized = cv2.resize(hand_roi, (IMG_SIZE, IMG_SIZE))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            normalized = gray / 255.0
            data.append(normalized.flatten())
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    np.save(f'dataset/{sign}.npy', np.array(data))
    print(f"✓ {sign} done! {len(data)} samples save ho gaye")
    print("5 second rest...")
    time.sleep(5)

cap.release()
cv2.destroyAllWindows()
print("\nSab signs collect ho gaye!")
