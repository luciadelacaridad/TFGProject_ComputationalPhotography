import numpy as np
import glob
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# cargamos los datos
pattern = (8,5)
imgpoints = np.load('prueba_6x9/puntos_imagen.npy')
imgpoints1 = imgpoints[0]

p1x, p1y = imgpoints1[0,0,0], imgpoints1[0,0,1]
p2x, p2y = imgpoints1[pattern[1]-1,0,0], imgpoints1[pattern[1]-1,0,1]
p3x, p3y = imgpoints1[35,0,0], imgpoints1[35,0,1]
p4x, p4y = imgpoints1[pattern[0]*pattern[1]-1,0,0], imgpoints1[pattern[0]*pattern[1]-1,0,1]
print(imgpoints1.shape)

