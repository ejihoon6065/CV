import numpy as np
import cv2
import random

def detect_face() :
    src = cv2.imread('kids.png')

    if src is None:
        print('Image load failed!')
        return

    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if classifier.empty():
        print('XML load failed!')
        return
    
    faces = classifier.detectMultiScale(src)

    for (x,y,w,h) in faces :
        cv2.rectangle(src, (x, y), (x+w, y+h), (255, 0, 255), 2)

    cv2.imshow('src', src)
    cv2.waitKey()
    cv2.destroyAllWindows()

# detect_face()


########################################################################
def detect_eyes() :
    src = cv2.imread('kids.png')

    if src is None:
        print('Image load failed!')
        return

    # xml 트레이닝을 통해 정교하게 만들 수 있음
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

    if face_classifier.empty() or eye_classifier.empty():
        print('XML load failed!')
        return
    
    faces = face_classifier.detectMultiScale(src)
    for (x1,y1,w1,h1) in faces :
        cv2.rectangle(src, (x1, y1), (x1+w1, y1+h1), (255, 0, 255), 2)

        faceROI = src[y1:y1 + h1, x1:x1 + w1]
        eyes = eye_classifier.detectMultiScale(faceROI)

        for (x2, y2, w2, h2) in eyes :
            center = (int(x2 + w2 / 2), int(y2 + h2 / 2))
            cv2.circle(faceROI, center, int(w2 / 5), (255, 0, 0), 2, cv2.LINE_AA)
                                     # int값 분모로 원의 크기 조절

    cv2.imshow('src', src)
    cv2.waitKey()
    cv2.destroyAllWindows()

detect_eyes()