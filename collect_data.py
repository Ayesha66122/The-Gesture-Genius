import cv2
import numpy as np
import os

SIGNS = ['hello', 'thankyou', 'yes', 'no', 'one', 'two', 'three', 'four', 'five']
SAMPLES_PER_SIGN = 200

os.makedirs('dataset', exist_ok=True)
cap = cv2.VideoCapture(0)

IMG_SIZE = 64

for sign in SIGNS:
    print(f"\n>>> '{sign}' ke liye tayar ho jao!")
    print("Haath camera ke BILKUL SAAMNE rakho")
    print("ENTER dabao shuru karne ke liye")
    input()

    data = []
    print("Collecting shuru...")

    while len(data) < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Sirf beech wala chota hissa lo
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        size = min(h, w) // 3
        x1 = cx - size
        x2 = cx + size
        y1 = cy - size
        y2 = cy + size

        hand_roi = frame[y1:y2, x1:x2]
        resized = cv2.resize(hand_roi, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        normalized = gray / 255.0
        data.append(normalized.flatten())

        print(f"{sign}: {len(data)}/{SAMPLES_PER_SIGN}", end='\r')

    np.save(f'dataset/{sign}.npy', np.array(data))
    print(f"\n✓ {sign} done!")
    print("5 second rest karo...")
    import time
    time.sleep(5)

cap.release()
print("\nSab signs collect ho gaye!")
