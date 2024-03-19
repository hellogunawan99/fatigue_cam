from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import mysql.connector
import sqlite3

# Function to initialize SQLite database and store data locally


def initialize_sqlite():
    conn_sqlite = sqlite3.connect('local_data.db')
    c = conn_sqlite.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS records
                 (unit TEXT, kondisi TEXT)''')
    conn_sqlite.commit()
    conn_sqlite.close()
# Function to write data to SQLite


def write_to_sqlite(unit, kondisi):
    conn_sqlite = sqlite3.connect('local_data.db')
    c = conn_sqlite.cursor()
    c.execute("INSERT INTO records (unit, kondisi) VALUES (?, ?)", (unit, kondisi))
    conn_sqlite.commit()
    conn_sqlite.close()


# check if the database is available
is_db_connected = False
try:
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="db_fatigue_cam"
    )
    mycursor = mydb.cursor()
    is_db_connected = True
except Exception as e:
    print("Error connecting to the database:", e)
    initialize_sqlite()  # Initialize SQLite if MySQL connection fails


# Function to read data from SQLite and send it to MySQL
def send_sqlite_data_to_mysql():
    conn_sqlite = sqlite3.connect('local_data.db')
    c = conn_sqlite.cursor()
    c.execute("SELECT * FROM records")
    rows = c.fetchall()
    for row in rows:
        unit, kondisi = row
        try:
            sql = "INSERT INTO record (unit, kondisi) VALUES (%s, %s)"
            val = (unit, kondisi)
            mycursor.execute(sql, val)
            mydb.commit()
            print("Data sent to MySQL:", val)
        except mysql.connector.Error as err:
            print("Error in MySQL operation:", err)
    # Clear SQLite table after sending data to MySQL
    c.execute("DELETE FROM records")
    conn_sqlite.commit()
    conn_sqlite.close()


# Set the path for the local file
local_data_path = "local_data.txt"


# Function to write data to a local file


def write_local_data(data):
    with open(local_data_path, "a") as file:
        file.write(data + "\n")


def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        print('pejam')
        s = 'espeak -a 200 -p 40 -s 150 -v id "{}"'.format(msg)
        os.system(s)

    if alarm_status2:
        print('menguap')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False


def eye_aspect_ratio(eye):
    # titik dari garis mata dari kiri ke kanan
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])  # titik garis mata atas ke bawah

    C = dist.euclidean(eye[0], eye[3])  # titik garis mata atas ke bawah

    ear = (A + B) / (2.0 * C)  # formula eye aspect ratio per mata

    return ear


def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0  # formula eye aspect ratio kedua mata
    return (ear, leftEye, rightEye)


def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.24
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 25  # sesuai jarak wajah wajar, karna untuk ngukur jarak bibir atas sama bawah
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

print("-> Loading the predictor and detector...")
detector = dlib.get_frontal_face_detector()
# detector = cv2.CascadeClassifier(
#    "haarcascade_frontalface_default.xml")  # Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
# vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
time.sleep(1.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    # rects = detector.detectMultiScale(gray, scaleFactor=1.1,
    #                                  minNeighbors=5, minSize=(30, 30),
    #                                  flags=cv2.CASCADE_SCALE_IMAGE)

    for rect in rects:
        # for (x, y, w, h) in rects:
        #    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Anda ngantuk!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if alarm_status == False:
                    alarm_status = True
                    t = Thread(target=alarm, args=('Bangun pak',))
                    t.deamon = True
                    t.start()
                    if is_db_connected:
                        try:
                            sql = "INSERT INTO record (unit, kondisi) VALUES (%s, %s)"
                            val = ("d3-555", "fatigue level 5")
                            mycursor.execute(sql, val)
                            mydb.commit()
                        except mysql.connector.Error as err:
                            print("Error in MySQL operation:", err)
                            write_to_sqlite("d3-555", "fatigue level 5")
                    else:
                        write_to_sqlite("d3-555", "fatigue level 5")
        else:
            COUNTER = 0
            alarm_status = False

        if (distance > YAWN_THRESH):
            cv2.putText(frame, "Anda lelah", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if alarm_status2 == False and saying == False:
                alarm_status2 = True
                t = Thread(target=alarm, args=('Istirahat sejenak pak',))
                t.deamon = True
                t.start()
                if is_db_connected:
                    try:
                        sql = "INSERT INTO record (unit, kondisi) VALUES (%s, %s)"
                        val = ("d3-555", "fatigue level 1")
                        mycursor.execute(sql, val)
                        mydb.commit()
                    except mysql.connector.Error as err:
                        print("Error in MySQL operation:", err)
                        write_to_sqlite("d3-555", "fatigue level 1")
                else:
                    write_to_sqlite("d3-555", "fatigue level 1")
        # Check if MySQL connection is reestablished and send data from SQLite
        if is_db_connected:
            send_sqlite_data_to_mysql()
        else:
            alarm_status2 = False

        cv2.putText(frame, "Pejam: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Menguap: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Fatigue Cam", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
