import cv2 as cv
import numpy as np

#crear una imagen en negro
img = np.zeros((512,512,3), np.uint8)

#dibuja una linea diagonal de grosor 5px
cv.line(img,(0,0),(511,511),(255,0,0),5)

cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv.circle(img,(447,63), 63, (0,0,255), -1)

cv.imshow('Black',img)
cv.waitKey(0)