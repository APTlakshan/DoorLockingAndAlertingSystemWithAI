import pandas as pd
import cv2
import urllib.request
import numpy as np
import os
from datetime import datetime
import face_recognition
import requests

path = r'E:\code\face_recognition\image_folder'
imgUrl = 'http://192.168.137.162/cam-hi.jpg'
okRqstUrl = 'http://192.168.137.154/?message=1'
noRqstUrl = 'http://192.168.137.154/?message=10'
authorized_dushan= 0
authorized_saman= 0
authorized_tharindu= 0

if 'Attendance.csv' in os.listdir(os.path.join(os.getcwd(), 'attendace')):
    print("There is an existing attendance file.")
    os.remove("Attendance.csv")
else:
    df = pd.DataFrame(list())
    df.to_csv("Attendance.csv")

images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    global authorized_dushan, authorized_saman, authorized_tharindu
    with open("Attendance.csv", 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            
            if name=="Tharindu":
                authorized_tharindu=1
            elif name=="Saman":
                authorized_saman=1
            elif name=="Dushan":
                authorized_dushan=1
            if authorized_dushan+authorized_saman+authorized_tharindu>= 3:
                f.writelines(',Door Opened')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

while True:
    img_resp = urllib.request.urlopen(imgUrl)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
        else:
            # Unrecognized face
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, 'Unrecognized', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        if authorized_dushan+authorized_saman+authorized_tharindu>= 3:
            print('Door Opened')
            response = requests.get(okRqstUrl)

            if response.status_code == 200:
                print("Response content:")
                print(response.text)
                authorized_dushan= 0
                authorized_saman= 0
                authorized_tharindu= 0
            else:
                print("Request failed with status code:", response.status_code)
                
        else:
            print('No')
            response = requests.get(noRqstUrl)

            if response.status_code == 200:
                print("Response content:")
                print(response.text)
            else:
                print("Request failed with status code:", response.status_code)

    cv2.imshow('Webcam', img)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cv2.destroyAllWindows()