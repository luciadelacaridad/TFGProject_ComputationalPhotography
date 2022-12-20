import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

#cargamos los datos

mtx = np.load('data/intrinsic_matrix.npy')
dist = np.load('data/distortion_coeffs.npy')
rvecs = np.load('data/rotation_vecs.npy')
tvecs = np.load('data/traslation_vecs.npy')
objpoints = np.load('data/puntos_objeto.npy')
imgpoints = np.load('data/puntos_imagen.npy')

#preparamos los datos para cada imagen

for i in range(len(objpoints)):

    imgpoints_ideal, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    imgpoints_ideal_new = imgpoints_ideal[:,0,:] #puntos imagen ideales array 42x2
    imgpoints_new = imgpoints[i,:,0,:] #puntos imagen medidos array 42x2
    np.save('data/imgpoints_ideal_rep'+str(i),imgpoints_ideal_new)
    np.save('data/imgpoints_medidos_rep'+str(i),imgpoints_new)

#imagen 1
id_1 = np.load('data/imgpoints_ideal_rep0.npy')
med_1 = np.load('data/imgpoints_medidos_rep0.npy')
#imagen 2
id_2 = np.load('data/imgpoints_ideal_rep1.npy')
med_2 = np.load('data/imgpoints_medidos_rep1.npy')
#imagen 3
id_3 = np.load('data/imgpoints_ideal_rep2.npy')
med_3 = np.load('data/imgpoints_medidos_rep2.npy')
#imagen 4
id_4 = np.load('data/imgpoints_ideal_rep3.npy')
med_4 = np.load('data/imgpoints_medidos_rep3.npy')
#imagen 5
id_5 = np.load('data/imgpoints_ideal_rep4.npy')
med_5 = np.load('data/imgpoints_medidos_rep4.npy')

#representacion
fig, ax = plt.subplots()
plt.scatter(med_1[:,0]-id_1[:,0], med_1[:,1]-id_1[:,1],marker='+',label='imagen1')
plt.scatter(med_2[:,0]-id_2[:,0], med_2[:,1]-id_2[:,1],marker='+',label='imagen2')
plt.scatter(med_3[:,0]-id_3[:,0], med_3[:,1]-id_3[:,1],marker='+',label='imagen3')
plt.scatter(med_4[:,0]-id_4[:,0], med_4[:,1]-id_4[:,1],marker='+',label='imagen4')
plt.scatter(med_5[:,0]-id_5[:,0], med_5[:,1]-id_5[:,1],marker='+',label='imagen5')
plt.xlim(-0.3,0.3)
ax.legend()
plt.show()