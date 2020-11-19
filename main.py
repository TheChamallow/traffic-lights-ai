import cv2
import numpy as np
from time import sleep

length_min=80 #Largeur minimale du rectangle
heigth_min=80 #Hauteur minimale du rectangle

offset=6 #Erreur permise entre les pixel  

pos_line=550 #Position de la ligne de comptage 

delay= 60 #FPS de la video

detec = []
cars= 0

	
def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('video.mp4')
subtraction = cv2.createBackgroundSubtractorMOG2()

while True:
    ret , frame1 = cap.read()
    time = float(1/delay)
    sleep(time) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtraction.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilated = cv2.morphologyEx (dilated, cv2. MORPH_CLOSE , kernel)
    contour,h=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, pos_line), (600, pos_line), (255,127,0), 3)
    
    for(i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_contour = (w >= length_min) and (h >= heigth_min)
        if not validate_contour:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        center = find_center(x, y, w, h)
        detec.append(center)
        cv2.circle(frame1, center, 4, (0, 0,255), -1)

        for (x,y) in detec:
            if y<(pos_line+offset) and y>(pos_line-offset) and x<606 and y>19:
                cars+=1
                cv2.line(frame1, (25, pos_line), (550, pos_line), (0,127,255), 3)  
                detec.remove((x,y))
                print("Car is detected : "+str(cars))        
       
    cv2.putText(frame1, "VEHICLE COUNT : "+str(cars), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)

    cv2.imshow("Video Originale" , frame1)
    cv2.imshow("Detecter",dilated)

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
