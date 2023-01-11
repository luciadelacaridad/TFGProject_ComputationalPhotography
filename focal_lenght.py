import numpy as np
import cv2 as cv



# matriz intrínseca K
K = np.load('prueba/intrinsic_matrix.npy')

# tamaño de las imágenes en píxeles
img = cv.cvtColor(cv.imread('prueba/calibrate_20221224_103930.png'), cv.COLOR_BGR2GRAY)
imageSize = img.shape[::-1]

# widht and height of the sensor in mm
apertureWidht = 2.46
apertureHeight = 1.8

# calculamos las caracteristicas

fovx, fovy, f, principalPoint, aspectRatio = cv.calibrationMatrixValues(K,imageSize,apertureWidht,apertureHeight)

mx = K[0,0] / f
my = K[1,1] / f

pixel_widht = 1 / mx
pixel_height = 1 / my


print(imageSize)
print('focal lenght in mm: ',f)
print('principal point in mm: ', principalPoint)
print('aspect ratio: ', aspectRatio)
print('mx en px/mm:',mx)
print('my en px/mm:',my)
print('anchura del pixel en mm:', pixel_widht)
print('altura del pixel en mm:', pixel_height)
