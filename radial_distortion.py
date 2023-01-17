# este programa dibuja líneas rectas entre puntos detectados en la imagen
# para apreciar la distorsión radial


import numpy as np
import cv2 as cv
import glob

# cargamos los datos

K = np.load('data/intrinsic_matrix.npy')
dist = np.load('data/distortion_coeffs.npy')
rvecs = np.load('data/rotation_vecs.npy')
tvecs = np.load('data/traslation_vecs.npy')
imgpoints_all = np.load('data/puntos_imagen.npy')
objpoints_all = np.load('data/puntos_objeto.npy')

# imágenes a usar
images = glob.glob('imagenes/*.png')

# dimensión del tablero
pattern = (9,6)

numero_imagen = 0 # contador

for i in range(len(images)):
    numero_imagen += 1

    # seleccionamos los ptos imagen de la imagen
    imgpoints = imgpoints_all[i]

    # definimos los 4 puntos de las esquinas del tablero
    p1x, p1y = imgpoints[0,0,0], imgpoints[0,0,1]
    p2x, p2y = imgpoints[pattern[1]-1,0,0], imgpoints[pattern[1]-1,0,1]
    p3x, p3y = imgpoints[pattern[0]*pattern[1]-pattern[1],0,0], imgpoints[pattern[0]*pattern[1]-pattern[1],0,1]
    p4x, p4y = imgpoints[pattern[0]*pattern[1]-1,0,0], imgpoints[pattern[0]*pattern[1]-1,0,1]

    # dibujamos las líneas sobre la imagen
    img = cv.imread(images[i])
    cv.line(img,(int(p1x),int(p1y)),(int(p2x),int(p2y)),(255,0,0),1)
    cv.line(img,(int(p2x),int(p2y)),(int(p4x),int(p4y)),(255,0,0),1)
    cv.line(img,(int(p3x),int(p3y)),(int(p4x),int(p4y)),(255,0,0),1)
    cv.line(img,(int(p3x),int(p3y)),(int(p1x),int(p1y)),(255,0,0),1)


    cv.imshow('imagen',img)
    cv.imwrite('distorsion_prueba/dibujo'+str(numero_imagen)+'.png',img)
    cv.waitKey()

