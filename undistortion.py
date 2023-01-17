# este programa corrige la distorsión de las imágenes
# dados los parámetros intrínsecos de la cámara utilizada

import numpy as np
import cv2 as cv
import glob

# cargamos los parámetros intrínsecos
K = np.load('data/intrinsic_matrix.npy')
dist = np.load('data/distortion_coeffs.npy')

# array con los paths de las imágenes que se van a utilizar
images = glob.glob('imagenes/*.png')

numero_imagen = 0
for fname in images:
    numero_imagen += 1
    img = cv.imread(fname)
    h,  w = img.shape[:2]
    newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))

    # corrige la imagen
    dst = cv.undistort(img, K, dist, None, newK)

    # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]

    # guardamos las imágenes con la distorsión corregida
    cv.imwrite('imagenes_corregidas/undist'+str(numero_imagen)+'.png', dst)

