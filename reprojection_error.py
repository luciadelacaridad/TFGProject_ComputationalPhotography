import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# cargamos los datos

mtx = np.load('data/intrinsic_matrix.npy')
dist = np.load('data/distortion_coeffs.npy')
rvecs = np.load('data/rotation_vecs.npy')
tvecs = np.load('data/traslation_vecs.npy')
objpoints = np.load('data/puntos_objeto.npy')
imgpoints = np.load('data/puntos_imagen.npy')

# error de retroproyeccion

sum_error = 0
error_vec = []
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    error_vec.append(error) # añade el error de retroproyección de cada imagen
    sum_error += error # suma el error

mean_error = sum_error / len(objpoints) #error de retroproyección medio

print(error_vec)
print('Mean re-projection error: ',mean_error)


#error de reroproyeccion barplot

x = np.arange(1,len(objpoints)+1) # array (1,...,11)

fig, ax = plt.subplots()
ax.bar(x, error_vec,color='violet')
plt.axhline(mean_error, color='limegreen', linestyle='--', label='Error de retroproyección medio')
ax.legend()
ax.set_title('Error de retroproyección para cada imagen')
ax.set_xlabel('Imagen')
ax.set_ylabel('Error de retroproyección')
ax.xaxis.set_major_locator(FixedLocator(np.arange(1,12)))
plt.show()