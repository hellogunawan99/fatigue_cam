import cv2
import numpy as np
import subprocess

mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if mouth_cascade.empty() or face_cascade.empty():
    raise IOError('Unable to load the cascade classifier xml file')

cap = cv2.VideoCapture(0)
ds_factor = 0.5

# Parameters
# Time threshold for triggering alarm when mouth is not detected (in seconds)
no_mouth_alarm_threshold = 3
start_time = None

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor,
                       fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        mouth_rects = mouth_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.7, minNeighbors=11)
        if len(mouth_rects) == 0:  # If mouth is not detected
            if start_time is None:
                start_time = cv2.getTickCount() / cv2.getTickFrequency()  # Start the timer
            else:
                elapsed_time = (cv2.getTickCount() /
                                cv2.getTickFrequency()) - start_time
                if elapsed_time >= no_mouth_alarm_threshold:
                    # cv2.putText(frame, "Mouth Covered", (50, 50),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print('Mulut Tertutup')
                    subprocess.Popen(
                        ["espeak", "-a", "200", "-p", "40", "-s", "150", "-v", "id", "Mulut Tertutup!"])
                    start_time = None  # Reset the timer after triggering the alarm
        else:  # If mouth is detected, reset the timer
            start_time = None

        for (mx, my, mw, mh) in mouth_rects:
            my += y
            cv2.rectangle(frame, (mx + x, my),
                          (mx + x + mw, my + mh), (0, 255, 0), 3)
            break

    cv2.imshow('Mouth Detector', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
