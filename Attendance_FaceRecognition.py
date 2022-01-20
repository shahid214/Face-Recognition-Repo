# Code By Shahid Akhtar

import face_recognition
import os
import numpy as np
import cv2
import pyttsx3

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voices', voices[0].id)

path = 'imagesAttendance'
mylist = os.listdir(path)

images = []
className = []

for cl in mylist:
    currentImg = cv2.imread(f'{path}/{cl}')
    images.append(currentImg)

    className.append(os.path.splitext(cl)[0])


# Let's encode each and every images in images  list


def findEncoding(imageslist):
    encodeList = []
    for img in imageslist:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


speak("Encoding has been initiated")

encodelistKnown = findEncoding(images)
speak("Encoding is complete!")

# Taking the test images through webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0

message_permission = False
while True:
    success, img = cap.read()

    CurrImg = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    CurrImg = cv2.cvtColor(CurrImg, cv2.COLOR_BGR2RGB)

    faceLocCurrFrame = face_recognition.face_locations(CurrImg)
    encodesCurrFrame = face_recognition.face_encodings(CurrImg, faceLocCurrFrame)

    for encodeFace, faceLoc in zip(encodesCurrFrame, faceLocCurrFrame):
        matches = face_recognition.compare_faces(encodelistKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodelistKnown, encodeFace)

        # print(faceDis)

        matchIndex = np.argmin(faceDis)

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if matches[matchIndex]:

            name = className[matchIndex].upper()

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            if count == 0:
                message = f'hello {name}, Verification successful'
                detection_status = "User Detected"
                print(detection_status + " : " + name)
                print("Attendance status: Present")

                message_permission = True

        else:
            name = "unknown"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        count = 1

    cv2.imshow('webcam', img)
    if message_permission:
        speak(detection_status)
        speak(message)
        message_permission = False

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

speak("Thank you! Have a great day")
cap.release()
cv2.destroyAllWindows()
