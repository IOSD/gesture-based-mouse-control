# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 19:44:55 2018

@author: Utsav Sharma
"""
import numpy as np
import cv2
from pynput.mouse import Button, Controller
import numpy as np   
import dlib  
from scipy.spatial import distance as dist 

PREDICTOR_PATH ="_path to shape_predictor_68_face_landmarks.dat file"  
   
FULL_POINTS = list(range(0, 68))  
FACE_POINTS = list(range(17, 68))  
   
JAWLINE_POINTS = list(range(0, 17))  
RIGHT_EYEBROW_POINTS = list(range(17, 22))  
LEFT_EYEBROW_POINTS = list(range(22, 27))  
NOSE_POINTS = list(range(27, 36))  
RIGHT_EYE_POINTS = list(range(36, 42))  
LEFT_EYE_POINTS = list(range(42, 48))  
MOUTH_OUTLINE_POINTS = list(range(48, 61))  
MOUTH_INNER_POINTS = list(range(61, 68))  
  
EYE_AR_THRESH = 0.25  
EYE_AR_CONSEC_FRAMES = 3  
  
COUNTER_LEFT = 0  
TOTAL_LEFT = 0  
  
COUNTER_RIGHT = 0  
TOTAL_RIGHT = 0  
def eye_aspect_ratio(eye):  
  # compute the euclidean distances between the two sets of  
  # vertical eye landmarks (x, y)-coordinates  
  A = dist.euclidean(eye[1], eye[5])  
  B = dist.euclidean(eye[2], eye[4])  
  
  # compute the euclidean distance between the horizontal  
  # eye landmark (x, y)-coordinates  
  C = dist.euclidean(eye[0], eye[3])  
  
  # compute the eye aspect ratio  
  ear = (A + B) / (2.0 * C)  
  
  # return the eye aspect ratio  
  return ear  
  
detector = dlib.get_frontal_face_detector()  
  
predictor = dlib.shape_predictor(PREDICTOR_PATH) 




mouse = Controller()
face_cascade = cv2.CascadeClassifier('path to haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('path to haarcascade_eye.xml')
#m = PyMouse()
#m.screen_size(1024,768)


cap = cv2.VideoCapture(0)

while 1:
    ################################
    ret, img = cap.read()  
 
    if ret:  
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
  
     rects = detector(gray, 0)  
     for rect in rects:  
       x = rect.left()  
       y = rect.top()  
       x1 = rect.right()  
       y1 = rect.bottom()  
  
       landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])  
  
       left_eye = landmarks[LEFT_EYE_POINTS]  
       right_eye = landmarks[RIGHT_EYE_POINTS]  
  
       left_eye_hull = cv2.convexHull(left_eye)  
       right_eye_hull = cv2.convexHull(right_eye)  
       cv2.drawContours(img, [left_eye_hull], -1, (0, 255, 0), 1)  
       cv2.drawContours(img, [right_eye_hull], -1, (0, 255, 0), 1)  
  
       ear_left = eye_aspect_ratio(left_eye)  
       ear_right = eye_aspect_ratio(right_eye)  
  
       cv2.putText(img, "E.A.R. Left : {:.2f}".format(ear_left), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  
       cv2.putText(img, "E.A.R. Right: {:.2f}".format(ear_right), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  
       if ear_left < EYE_AR_THRESH:  
         COUNTER_LEFT += 1  
       else:  
         if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:  
           TOTAL_LEFT += 1  
           print("Left eye winked")
           mouse.click(Button.left, 1)
         COUNTER_LEFT = 0
        
       if ear_right < EYE_AR_THRESH:  
         COUNTER_RIGHT += 1  
       else:  
         if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:  
           TOTAL_RIGHT += 1  
           print("Right eye winked")
           mouse.click(Button.right, 1)
         COUNTER_RIGHT = 0
     cv2.putText(img, "Wink Left : {}".format(TOTAL_LEFT), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  
     cv2.putText(img, "Wink Right: {}".format(TOTAL_RIGHT), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  
  
    ################################
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
       
        
        #m.move(200, 200)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
           
            mouse.position = (x, y)

    cv2.imshow('img',img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
      
      cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
   
