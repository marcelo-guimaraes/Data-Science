import cv2 as cv
import numpy as np

img = cv.imread('box.png', 0)
img2 = cv.imread('box_in_scene.png', 0)

orb = cv.ORB_create()

kp1, des1 = orb.detectAndCompute(img,None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf= cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
matches = bf.match (des1, des2)

matches = sorted(matches, key = lambda x:x.distance)
img3 = cv.drawMatches (img, kp1, img2, kp2, matches [:10], None, flags = 2)

cv.imshow('Janela',img3)
cv.waitKey(0)
cv.destroyAllWindows()
