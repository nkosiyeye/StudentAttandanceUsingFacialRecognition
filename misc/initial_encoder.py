import cv2
import pickle
import face_recognition
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("C:\\Users\\yenzi\\Downloads\\Face-Recognition\\Face-Recognition-System-for-Student-Attendance-main\\serviceAccountKey.json")
firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": "https://todoapp-63646-default-rtdb.firebaseio.com/",
        # database URL
        "storageBucket": "todoapp-63646.firebasestorage.app",
    },
)

# student images
folderPath = "Face-Recognition-System-for-Student-Attendance-main/static/Files/Images"
imgPathList = os.listdir(folderPath)
print(imgPathList)
imgList = []
studentIDs = []

for path in imgPathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    # print(os.path.splitext(path)[0])
    studentIDs.append(os.path.splitext(path)[0])

    fileName = f"{folderPath}/{path}"
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

print(studentIDs)


def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]

        encodeList.append(encode)

    return encodeList


print("Encoding Started")

encodeListKnown = findEncodings(imgList)

encodeListKnownWithIds = [encodeListKnown, studentIDs]

file = open("EncodeFile.p", "wb")
pickle.dump(encodeListKnownWithIds, file)
file.close()

# print(encodeListKnown)

print("Encoding Ended")
