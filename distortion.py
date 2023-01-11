import numpy as np
import cv2 as cv
import glob

# cargamos los datos

K = np.load('prueba_6x9/intrinsic_matrix.npy')
dist = np.load('prueba_6x9/distortion_coeffs.npy')
rvecs = np.load('prueba_6x9/rotation_vecs.npy')
tvecs = np.load('prueba_6x9/traslation_vecs.npy')
imgpoints_all = np.load('prueba_6x9/puntos_imagen.npy')
objpoints_all = np.load('prueba_6x9/puntos_objeto.npy')

# imágenes a usar

images = glob.glob('prueba_6x9/*.jpg')

# dimensión del tablero

pattern = (8,5)


for i in range(len(images)):

    # seleccionamos los parametros extrínsecos y los ptos imagen de la imagen

    imgpoints = imgpoints_all[i]

    # definimos los 4 puntos de las esquinas del tablero
    p1x, p1y = imgpoints[0,0,0], imgpoints[0,0,1]
    p2x, p2y = imgpoints[4,0,0], imgpoints[4,0,1]
    p3x, p3y = imgpoints[35,0,0], imgpoints[35,0,1]
    p4x, p4y = imgpoints[39,0,0], imgpoints[39,0,1]

    # dibujamos las líneas sobre la imagen
    img = cv.imread(images[i])
    cv.line(img,(int(p1x),int(p1y)),(int(p2x),int(p2y)),(255,0,0),2)
    cv.line(img,(int(p2x),int(p2y)),(int(p4x),int(p4y)),(255,0,0),2)
    cv.line(img,(int(p3x),int(p3y)),(int(p4x),int(p4y)),(255,0,0),2)
    cv.line(img,(int(p3x),int(p3y)),(int(p1x),int(p1y)),(255,0,0),2)


    cv.imshow('imagen',img)
    cv.waitKey()

