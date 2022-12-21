import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# cargamos los datos

mtx = np.load('data/intrinsic_matrix.npy')
dist = np.load('data/distortion_coeffs.npy')
rvecs = np.load('data/rotation_vecs.npy')
tvecs = np.load('data/traslation_vecs.npy')
objpoints = np.load('data/puntos_objeto.npy')
imgpoints = np.load('data/puntos_imagen.npy')

# preparamos los datos para cada imagen y los representamos

fig, ax = plt.subplots()

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    imgpoints_ideal = imgpoints2[:,0,:] # puntos imagen ideales array 42x2
    imgpoints_med = imgpoints[i,:,0,:] # puntos imagen medidos array 42x2
    delta_x = imgpoints_med[:,0] - imgpoints_ideal[:,0]
    delta_y = imgpoints_med[:,1] - imgpoints_ideal[:,1]
    plt.scatter(delta_x,delta_y,marker='+',label='imagen'+str(i+1))

plt.xlim(-0.4,0.4)
plt.ylim(-0.4,0.4)
ax.legend()
plt.show()