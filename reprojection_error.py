import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,IndexLocator

# cargamos los datos

mtx = np.load('data/intrinsic_matrix.npy')
dist = np.load('data/distortion_coeffs.npy')
rvecs = np.load('data/rotation_vecs.npy')
tvecs = np.load('data/traslation_vecs.npy')
objpoints = np.load('data/puntos_objeto.npy')
imgpoints = np.load('data/puntos_imagen.npy')
error_RMS = np.load('data/errors.npy') # error de retroproyección RMS para cada imagen
mean_error_RMS = np.load('data/mean_error.npy')

print('Mean RMS re-projection error: ',mean_error_RMS)



#error de retroproyeccion barplot

x = np.arange(1,len(objpoints)+1) # array (1,...,N) N = número de imágenes

fig, ax = plt.subplots()
ax.bar(x, error_RMS[:,0], color='violet', label='Error RMS')
plt.axhline(mean_error_RMS, color='red', linestyle='--', label='Error de retroproyección RMS medio')
ax.legend()
ax.set_title('Error de retroproyección RMS para cada imagen')
ax.set_xlabel('Imagen')
ax.set_ylabel('Error de retroproyección RMS (px)')
ax.xaxis.set_major_locator(FixedLocator(np.arange(1,len(objpoints)+1)))
ax.yaxis.set_minor_locator(FixedLocator(mean_error_RMS))
plt.show()