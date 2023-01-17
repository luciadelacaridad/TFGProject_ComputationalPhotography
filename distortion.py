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

images = glob.glob('imagenes_corregidas/*.png')

# dimensión del tablero

pattern = (9,6)
numero_imagen = 0
for i in range(len(images)):
    numero_imagen+=1
    # seleccionamos los parametros extrínsecos y los ptos imagen de la imagen

    imgpoints = imgpoints_all[i]

    # definimos los 4 puntos de las esquinas del tablero
    p1x, p1y = imgpoints[0,0,0], imgpoints[0,0,1]
    p2x, p2y = imgpoints[5,0,0], imgpoints[5,0,1]
    p3x, p3y = imgpoints[48,0,0], imgpoints[48,0,1]
    p4x, p4y = imgpoints[53,0,0], imgpoints[53,0,1]

    # dibujamos las líneas sobre la imagen
    img = cv.imread(images[i])
    cv.line(img,(int(p1x),int(p1y)),(int(p2x),int(p2y)),(255,0,0),1)
    cv.line(img,(int(p2x),int(p2y)),(int(p4x),int(p4y)),(255,0,0),1)
    cv.line(img,(int(p3x),int(p3y)),(int(p4x),int(p4y)),(255,0,0),1)
    cv.line(img,(int(p3x),int(p3y)),(int(p1x),int(p1y)),(255,0,0),1)


    cv.imshow('imagen',img)
    cv.imwrite('distorsion_prueba/dibujo_corregida'+str(numero_imagen)+'.png',img)
    cv.waitKey()

