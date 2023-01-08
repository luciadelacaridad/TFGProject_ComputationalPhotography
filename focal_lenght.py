import numpy as np
import cv2 as cv



# matriz intrínseca K
K = np.load('data2/intrinsic_matrix.npy')

# tamaño de las imágenes en píxeles
img = cv.cvtColor(cv.imread('prueba/calibrate_20221224_103930.png'), cv.COLOR_BGR2GRAY)
imageSize = img.shape[::-1]

# widht and height of the sensor in mm
apertureWidht = 5
apertureHeight = 5

# calculamos las caracteristicas

fovx, fovy, focalLenght, principalPoint, aspectRatio = cv.calibrationMatrixValues(K,imageSize,apertureWidht,apertureHeight)

print('focal lenght in mm: ',focalLenght)
print('principal point in mm: ', principalPoint)
print('aspect ratio: ', aspectRatio)

