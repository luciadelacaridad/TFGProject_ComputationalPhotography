import numpy as np
import glob
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# cargamos los datos

K = np.load('data2/intrinsic_matrix.npy')
dist = np.load('data2/distortion_coeffs.npy')
rvecs = np.load('data2/rotation_vecs.npy')
tvecs = np.load('data2/traslation_vecs.npy')
objpoints = np.load('data2/puntos_objeto.npy')
imgpoints = np.load('data2/puntos_imagen.npy')
perViewErrors = np.load('data2/errors.npy')

img = cv.cvtColor( cv.imread('Resources/left01.jpg'), cv.COLOR_BGR2GRAY)
print()


