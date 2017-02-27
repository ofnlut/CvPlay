import numpy as np
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = vars(ap.parse_args())

#load up our cascades
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")

#read and convert our imgage to grayscale
img = cv2.imread(args["image"])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh), (0,255,0),2)

cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()