import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# cargamos los datos

K = np.load('data/intrinsic_matrix.npy')
dist = np.load('data/distortion_coeffs.npy')
rvecs = np.load('data/rotation_vecs.npy')
tvecs = np.load('data/traslation_vecs.npy')
objpoints = np.load('data/puntos_objeto.npy')
imgpoints = np.load('data/puntos_imagen.npy')

# preparamos los datos para cada imagen y los representamos

fig, ax1 = plt.subplots()
error_x = []
error_y = []

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    imgpoints_ideal = imgpoints2[:,0,:] # puntos imagen ideales array 42x2
    imgpoints_med = imgpoints[i,:,0,:] # puntos imagen medidos array 42x2
    delta_x = imgpoints_med[:,0] - imgpoints_ideal[:,0]
    delta_y = imgpoints_med[:,1] - imgpoints_ideal[:,1]
    error_x.append(delta_x)
    error_y.append(delta_y)
    ax1.plot(delta_x,delta_y,marker='+',label='imagen'+str(i+1),ls='')


# ax1.legend()
plt.title('Error de cada de cada punto')
plt.xlabel('Error en el eje $u$ en px')
plt.ylabel('Error en el eje $v$ en px')
plt.show()

# guardamos los datos
np.save('data/delta_x_list',error_x)
np.save('data/delta_y_list',error_y)


