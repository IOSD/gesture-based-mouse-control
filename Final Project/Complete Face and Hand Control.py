# -*- coding: utf-8 -*-

"""
Created on Fri Jul 27 19:44:55 2018

@author: Utsav & Shubham
"""
import numpy as np
import cv2
import wx
from pynput.mouse import Button, Controller
import numpy as np   
import dlib  
from scipy.spatial import distance as dist 
j=1
app = wx.App(False)   #creating a object app. but we have to delete it in the end.

(sx,sy)=wx.GetDisplaySize() #screen size
(camx,camy)=(640,480)   #image resolution.
lowerBound=np.array([33,80,40])
upperBound=np.array([102,255,255])
kernalopen=np.ones([5,5])
kernalclose=np.ones([20,20])

cap=cv2.VideoCapture(0)
cap.set(3,camx)
cap.set(4,camy)

mlocold=np.array([0,0])
mlocnew=np.array([0,0])
df=5
#mlocold=mlocold+(target-mlocold)//df

pinch=0
ox,oy,ow,oh=(0,0,0,0)
PREDICTOR_PATH ="C:\\Users\\Tony Stark\\Desktop\\blink-detection\\shape_predictor_68_face_landmarks.dat"  
   
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
face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')
#m = PyMouse()
#m.screen_size(1024,768)


cap = cv2.VideoCapture("D:\\TwitchDownloads\\3dGameDev.mp4")

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
       cv2.drawContours(img, [left_eye_hull], -1, (0, 255, 255), 1)  
       cv2.drawContours(img, [right_eye_hull], -1, (0, 255, 255), 1)  
  
       ear_left = eye_aspect_ratio(left_eye)  
       ear_right = eye_aspect_ratio(right_eye)  
  
       #cv2.putText(img, "E.A.R. Left : {:.2f}".format(ear_left), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  
       #cv2.putText(img, "E.A.R. Right: {:.2f}".format(ear_right), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  
       if ear_left < EYE_AR_THRESH:  
         COUNTER_LEFT += 1  
       else:  
         if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:  
           TOTAL_LEFT += 1  
           print("Left eye winked")
           cv2.putText(img, "Left Click ".format(TOTAL_LEFT), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
           mouse.click(Button.left, 1)
         COUNTER_LEFT = 0
        
       if ear_right < EYE_AR_THRESH:  
         COUNTER_RIGHT += 1  
       else:  
         if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:  
           TOTAL_RIGHT += 1  
           print("Right eye winked")
           cv2.putText(img, "Right Click".format(TOTAL_RIGHT), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
           mouse.click(Button.right, 1)
         COUNTER_RIGHT = 0
       
       
  
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
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
            
    ###########################
    hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
    mask =cv2.inRange(hsv,lowerBound,upperBound)
    
    maskopen =cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernalopen) #opening will remove all the dots randomly popping here.
    maskclose =cv2.morphologyEx(maskopen,cv2.MORPH_CLOSE,kernalclose)#closing will close the small holes that are present in the actual object
    
    maskfinal=maskclose
    _,cont,h=cv2.findContours(maskfinal.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    

    if(len(cont)==2):
    
        if(pinch==1):
            pinch=0
            pass
        
        x1,y1,w1,h1=cv2.boundingRect(cont[0])
        x2,y2,w2,h2=cv2.boundingRect(cont[1])
        cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
        cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
        cx1=x1+w1//2  # x coordinate of centre of cont[0]. 
        cy1=y1+h1//2  # y coordinate of centre of cont[0].
        cx2=x2+w2//2
        cy2=y2+h2//2
        cx=(cx1+cx2)//2 # x coordinate of center of line
        cy=(cy1+cy2)//2 # y coordinate of center of line.
        l=list()
        m=list()
        l.append(cx)
        m.append(cy)
        
        cv2.line(img,(cx1,cy1),(cx2,cy2),(255,0,0),2)

        cv2.circle(img,(cx,cy),2,(0,0,255),2)
        mlocnew=mlocold+(((cx,cy)-mlocold)//df) 
        mouse.position=(sx-(mlocnew[0]*sx//camx),(mlocnew[1]*sy//camy))
        while (mouse.position!=(sx-(mlocnew[0]*sx//camx),(mlocnew[1]*sy//camy))):
            pass
        mlocold=mlocnew
        
        ox,oy,ow,oh=cv2.boundingRect(np.array([[[x1,y1],[x1+w1,y1+h1],[x2,y2],[x2+w2,y2+h2]]]))
        
        #cv2.rectangle(img,(ox,oy),(ox+ow,oy+oh),(255,0,0),2)
        br=ow*oh
    elif(len(cont)==1):
        x,y,w,h=cv2.boundingRect(cont[0])
        sr=w*h
        br=ow*oh
        

        if(pinch==0):
            mr=(sr-br)*100
            if(abs(mr/sr)<20):
                pinch=1
                mouse.click(Button.left,2)
                cv2.putText(img, "Left Click ".format(TOTAL_LEFT), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                ox,oy,ow,oh=(0,0,0,0)
        
        x,y,w,h=cv2.boundingRect(cont[0])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cx=x+w//2
        cy=y+h//2
        cv2.circle(img,(cx,cy),(w+h)//4,(0,0,255),2)
        
        mlocnew=mlocold+((cx,cy)-mlocold)//df
        mouse.position=(sx-(mlocnew[0]*sx//camx),(mlocnew[1]*sy//camy))
    
        while (mouse.position!=(sx-(mlocnew[0]*sx//camx),(mlocnew[1]*sy//camy))):
            pass
        mlocold=mlocnew
        #if(a3>a1 and a3<(a1+a2)):
            #mouse.click(Button.left,1)
    
    #cv2.imshow("mask",mask)
    #cv2.imshow("maskopen",maskopen)
    #cv2.imshow("maskclose",maskclose)
    #cv2.imshow("maskfinal",maskfinal)
    ###########################
    cv2.imshow('img',img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
      
      cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
   