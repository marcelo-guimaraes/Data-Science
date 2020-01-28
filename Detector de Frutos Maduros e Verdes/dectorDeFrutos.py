from collections import deque
import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX


greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

#yellowLower = (0, 100, 100)
#yellowUpper = (10, 255, 255)
#yellowLower = (0, 157, 138)
#yellowUpper = (82, 255, 255)
sensitivity = 15
lower_red_0 = (0, 100, 100) 
upper_red_0 = (sensitivity, 255, 255)
lower_red_1 = (180 - sensitivity, 100, 100) 
upper_red_1 = (180, 255, 255)


camera = cv2.VideoCapture(0)

i=0

while True:
    (grabbed, frame) = camera.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    maskG = cv2.inRange(hsv, greenLower, greenUpper)
    maskG = cv2.erode(maskG, None, iterations=2)
    maskG = cv2.dilate(maskG, None, iterations=2)


    mask_0 = cv2.inRange(hsv, lower_red_0 , upper_red_0)
    mask_1 = cv2.inRange(hsv, lower_red_1 , upper_red_1 )
    maskY = cv2.bitwise_or(mask_0, mask_1)
    
    #maskY = cv2.inRange(hsv, yellowLower, yellowUpper)
    maskY = cv2.erode(maskY, None, iterations=5)
    maskY = cv2.dilate(maskY, None, iterations=5)

    cntsG = cv2.findContours(maskG.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cntsY = cv2.findContours(maskY.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cntsG) > 0:
        c = max(cntsG, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 50:
            cv2.putText(frame,'Fruto Verde',(30+i,250), font, .5,(30, 20, 160),2,cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    if len(cntsY) > 0:
        c = max(cntsY, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 5:
            cv2.putText(frame,'Fruto Maduro',(30+i,300), font, .5,(30, 170, 255),2,cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            

    cv2.imshow("Frame", frame)
    cv2.imshow("MaskG", maskG)
    cv2.imshow("MaskY", maskY)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
