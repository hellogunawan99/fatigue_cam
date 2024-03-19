import cv2
import os
from threading import Thread


def alarm(msg):
    global alarm_status

    while alarm_status:
        print('pejam')
        s = 'espeak "'+msg+'"'
        os.system(s)


# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Flag to indicate if face is blocked
face_blocked = False

# Main loop
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(5, 5))

    # Check if face is detected
    if len(faces) == 0:
        # If no face is detected, set face_blocked flag to True
        face_blocked = False
    else:
        # If face is detected, set face_blocked flag to False
        face_blocked = True

    # If face is blocked, play the alarm sound
    if face_blocked == False:
        alarm_status = True
        t = Thread(target=alarm, args=('Tertutup',))
        t.deamon = True
        t.start()

        # Optionally, you can add other actions here like sending a notification

    # Display the frame
    cv2.imshow('Face Block Detector', frame)

    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
