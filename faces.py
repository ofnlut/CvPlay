import numpy as np
import cv2
import argparse
import urllib.request


ap = argparse.ArgumentParser()
ap.add_argument("--foo", help='foo help')
ap.add_argument("-i", "--image", required=False, help="Path to image")
ap.add_argument("-url", "--imgurl", required=False, help="Url link to image")
ap.add_argument("-v", "--video", required=False, help="video or no?")
args = vars(ap.parse_args())

#load up our cascades
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")


#read and convert our imgage to grayscale
if args["image"]:

    img = cv2.imread("images/"+args["image"])
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

if args["imgurl"]:
    response = urllib.request.urlopen(args["imgurl"])
    translate = np.array(bytearray(response.read()),dtype=np.uint8)
    translate = cv2.imdecode(translate, cv2.IMREAD_COLOR)

    img = cv2.imread(translate)
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


if args["video"]:
    vid = cv2.VideoCapture(int(args["video"]))

    while True:
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh), (0,255,0),2)
        cv2.imshow('video',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
