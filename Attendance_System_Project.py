import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = "ImageLibrary"
images = []
names = []
myList = os.listdir(path)
# print(myList)

for cl in myList:
    imgCurr = cv2.imread(f'{path}/{cl}')
    images.append(imgCurr)
    names.append(os.path.splitext(cl)[0])


def calculateEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def MarkAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')


print("Start Encoding.....")
encodedListKnown = calculateEncodings(images)
print("Encoding complete .........")


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(imgSmall)
    encodeCurrentFrame = face_recognition.face_encodings(imgSmall, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodeCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodedListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodedListKnown, encodeFace)
        # print(matches)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = names[matchIndex].upper()
            # print(name)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            MarkAttendance(name)

        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            name = "UNKNOWN"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
