import requests
import numpy as np
import cv2 as cv
import face_recognition



url = "http://192.168.1.6:8080/shot.jpg"

while True:
   
    RawData = requests.get(url, verify=False)

    
    One_D_Arry = np.array(bytearray(RawData.content),dtype = np.uint8)
    

    frame = cv.imdecode(One_D_Arry, -1)

    rgb_frame= frame[:,:, ::-1]
    face_location = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame,face_location)

    for(top,right,bottom,left,), face_location in zip (face_location,face_encodings):
        cv.rectangle(frame,(left,top), (right,bottom),(0,0,255),2)

    cv.imshow("Windows", frame)

    if cv.waitKey(1)==ord('q'):
        break
