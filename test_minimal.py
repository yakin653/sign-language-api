import cv2
print("✅ OpenCV OK")

import mediapipe as mp
print("✅ MediaPipe OK")

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Caméra OK")
    ret, frame = cap.read()
    if ret:
        print("✅ Image capturée")
    cap.release()
else:
    print("❌ Caméra non trouvée")