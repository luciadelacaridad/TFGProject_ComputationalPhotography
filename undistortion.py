import numpy as np
import cv2 as cv
import glob

# cargamos los parámetros intrínsecos

K = np.load('prueba_6x9/intrinsic_matrix.npy')
dist = np.load('prueba_6x9/distortion_coeffs.npy')

# array con los paths de las imágenes que se van a utilizar

images = glob.glob('prueba_6x9/*.jpg')

for fname in images:
    img = cv.imread(fname)
    h,  w = img.shape[:2]
    newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv.undistort(img, K, dist, None, newK)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # guardamos las imágenes con la distorsión corregida
    cv.imwrite(fname.replace('Pro','Undist'), dst)

