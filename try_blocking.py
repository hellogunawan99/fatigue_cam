import cv2
import time
import subprocess

# Function to detect mouth


def detect_mouth(frame, face_cascade, mouth_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)
        for (mx, my, mw, mh) in mouth:
            return True
    return False


# Load cascade classifiers
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_mouth.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Parameters
block_time_threshold = 3  # Time threshold for mouth blocking in seconds
start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect mouth
    mouth_detected = detect_mouth(frame, face_cascade, mouth_cascade)

    # If mouth is detected
    if mouth_detected:
        if start_time is None:
            start_time = time.time()  # Start counting time
        else:
            elapsed_time = time.time() - start_time
            if elapsed_time >= block_time_threshold:
                print("Mouth covered for too long! Triggering alarm.")
                # Use espeak to give voice alarm
                subprocess.Popen(
                    ["espeak", "-ven+f3", "-s150", "Mouth covered for too long!"])
                break
    else:
        start_time = None  # Reset the timer if mouth is not detected

    cv2.imshow('Mouth Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
