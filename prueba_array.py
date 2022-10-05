import numpy as np
import cv2 as cv
import glob

images = glob.glob('Resources\*.jpg')
img1 = images[:0]
img12 = images[:2]
imgtotal = images[:len(images)]
print(img1)
print(len(images))
print(np.arange(1,len(images)))