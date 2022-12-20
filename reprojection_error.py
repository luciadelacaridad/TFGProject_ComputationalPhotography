import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

#cargamos los datos

mtx = np.load('data/intrinsic_matrix.npy')
dist = np.load('data/distortion_coeffs.npy')
rvecs = np.load('data/rotation_vecs.npy')
tvecs = np.load('data/traslation_vecs.npy')
objpoints = np.load('data/puntos_objeto.npy')
imgpoints = np.load('data/puntos_imagen.npy')

#error de retroproyeccion

sum_error = 0
error_vec = []
for i in range(len(objpoints)):
    error = 0
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    error_vec.append(error)
    sum_error += error

mean_error = sum_error / len(objpoints)
print(error_vec) #error para cada imagen
print('Mean reprojection error: ',mean_error) #error medio


#error de reroproyeccion barplot

x = np.arange(1,len(objpoints)+1)

fig, ax = plt.subplots()
ax.bar(x, error_vec)
plt.axhline(mean_error, color='r', linestyle='--', label='Error de retroproyección medio')
ax.legend()
ax.set_title('Figura1')
ax.set_xlabel('Imagen')
ax.set_ylabel('Error de reproyección')
ax.xaxis.set_major_locator(FixedLocator(np.arange(1,12)))
plt.show()