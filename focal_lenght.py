# este programa calcula la distancia focal de la cámara sabiendo su matriz intrínseca K
#

import numpy as np
import cv2 as cv


# matriz intrínseca K
K = np.load('data/intrinsic_matrix.npy')

# tamaño de las imágenes en píxeles
imageSize = (640,480)

# widht and height of the sensor in mm (1/6'')
apertureWidht = 2.40
apertureHeight = 1.80

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
