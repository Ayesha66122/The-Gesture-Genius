import cv2
import numpy as np
import os
import time

SIGNS = ['hello', 'thankyou', 'yes', 'no', 'one', 'two', 'three', 'four', 'five']
SAMPLES_PER_SIGN = 200

os.makedirs('dataset', exist_ok=True)

# Different backend try karo
for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        break

IMG_SIZE = 64

for sign in SIGNS:
    print(f"\n>>> '{sign}' ke liye tayar ho jao!")
    print("Window mein GREEN BOX dikhega — haath wahan rakho")
    print("ENTER dabao shuru karne ke liye")
    input()

    data = []
    collecting = False
    start_time = time.time()

    while len(data) < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        size = min(h, w) // 3
        x1 = cx - size
        x2 = cx + size
        y1 = cy - size
        y2 = cy + size

        # Green box dikhao
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Text dikhao
        cv2.putText(frame, f"Sign: {sign.upper()}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Samples: {len(data)}/{SAMPLES_PER_SIGN}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE=Start/Stop  Q=Quit", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        if collecting:
            hand_roi = frame[y1:y2, x1:x2]
            resized = cv2.resize(hand_roi, (IMG_SIZE, IMG_SIZE))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            normalized = gray / 255.0
            data.append(normalized.flatten())
            cv2.putText(frame, "RECORDING...", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        try:
            cv2.imshow('Gesture Genius - Data Collector', frame)
        except:
            pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            collecting = not collecting
        elif key == ord('q'):
            break

    np.save(f'dataset/{sign}.npy', np.array(data))
    print(f"✓ {sign} done! {len(data)} samples save ho gaye")
    cv2.destroyAllWindows()
    print("5 second rest...")
    time.sleep(5)

cap.release()
cv2.destroyAllWindows()
print("\nSab signs collect ho gaye!")
