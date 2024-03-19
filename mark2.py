import cv2
import numpy as np
import subprocess

# Load cascade classifiers
mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if the classifiers are loaded successfully
if mouth_cascade.empty() or face_cascade.empty():
    raise IOError('Unable to load the cascade classifier xml file')

# Initialize video capture
cap = cv2.VideoCapture(0)
ds_factor = 0.5

# Parameters
no_mouth_alarm_threshold = 3
start_time = None

# Loop for video processing
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor,
                       fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        # Adjust parameters for mouth detection
        mouth_rects = mouth_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.2, minNeighbors=8, minSize=(20, 20))
        # If mouth is not detected
        if len(mouth_rects) == 0:
            if start_time is None:
                start_time = cv2.getTickCount() / cv2.getTickFrequency()
            else:
                elapsed_time = (cv2.getTickCount() /
                                cv2.getTickFrequency()) - start_time
                if elapsed_time >= no_mouth_alarm_threshold:
                    print('Mouth Covered')
                    subprocess.Popen(
                        ["espeak", "-a", "200", "-p", "40", "-s", "150", "-v", "id", "Mulut Tertutup!"])
                    start_time = None
        else:  # If mouth is detected, reset the timer
            start_time = None

        # Draw rectangles around detected mouths
        for (mx, my, mw, mh) in mouth_rects:
            my += y
            cv2.rectangle(frame, (mx + x, my),
                          (mx + x + mw, my + mh), (0, 255, 0), 3)
            break  # Process only the first detected mouth

    cv2.imshow('Mouth Detector', frame)

    # Exit loop on 'ESC' key press
    c = cv2.waitKey(1)
    if c == 27:
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
