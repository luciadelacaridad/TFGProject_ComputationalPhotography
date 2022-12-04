import numpy as np
import cv2 as cv
import glob

img = cv.imread('Resources\left01.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(gray.shape)