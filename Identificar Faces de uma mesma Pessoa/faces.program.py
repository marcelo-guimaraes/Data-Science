import cv2 as cv
import numpy as np
import glob
from skimage.feature import local_binary_pattern

def image2feature (filename):
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(img, 8*3, 3, method = 'uniform')
    x = np.unique(lbp.ravel(), return_counts = True)
    feature = x[1]/sum(x[1])
    return feature

def distance (x,y):
    return np.linalg.norm(x-y)

query = 'Faces/Test/15.png'
queryFeature = image2feature(query)

menorDistancia = 99999999
mostSimilar = ""
for filename in glob.iglob('Faces/Train/*.png'):
    databaseFeature = image2feature(filename)
    distancia = distance (queryFeature, databaseFeature)
    if (distancia < menorDistancia):
        menorDistancia = distancia
        mostSimilar = filename

img = cv.imread(mostSimilar, 0)
query = cv.imread(query, 0)
                                

cv.imshow("AOw",img)
cv.imshow("AO",query)
cv.waitKey(0)
cv.destroyAllWindows()
