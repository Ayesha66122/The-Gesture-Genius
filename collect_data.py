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
    print("ENTER dabao shuru karne ke liye")
    input()  # ENTER dabao

    data = []

    while len(data) < SAMPLES_PER_SIGN:
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
        data.append(normalized.flatten())

        print(f"{sign}: {len(data)}/{SAMPLES_PER_SIGN}", end='\r')

    np.save(f'dataset/{sign}.npy', np.array(data))
    print(f"\n✓ {sign} done! {len(data)} samples save ho gaye")

cap.release()
print("\nSab signs collect ho gaye!")
