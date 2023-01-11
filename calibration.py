import numpy as np
import cv2 as cv
import glob


# dimension del chessboard
chessboard = (8,5)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# preparamos los puntos objeto
objp = np.zeros((chessboard[0]*chessboard[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard[1],0:chessboard[0]].T.reshape(-1,2) # puntos de la forma (0,0,0),(1,0,0),(2,0,0),etc


# arrays vacíos para almacenar los puntos objeto y los puntos imagen
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

# array con los paths de las imágenes que se van a utilizar
images = glob.glob('prueba_6x9/*.jpg')

aptas = 0  # contador para ver cuantos patterns se detectan

for fname in images:        # bucle para cada imagen
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # pasa la imagen a blanco y negro
    ret, corners = cv.findChessboardCorners(gray, (chessboard[1],chessboard[0]), None) # esta funcion encuentra las esquinas del chessboard
    if ret == True:  # si el pattern ha sido detectado
        aptas += 1
        objpoints.append(objp) # añade puntos objeto
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria) # esta funcion refina el colocamiento de las esquinas
        imgpoints.append(corners2) # añade los puntos imagen refinados
        cv.drawChessboardCorners(img, (chessboard[1],chessboard[0]), corners2, ret) # dibuja los puntos en el chessboard
        # cv.imshow('img', img)
        # cv.waitKey()
    else: print('error')
cv.destroyAllWindows()

# obtenemos la camera matrix, distortion coeffs, rotation and traslation vectors

ret, K, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors= cv.calibrateCameraExtended(objpoints, imgpoints, gray.shape[::-1], None, None)

# error de retroproyección RMS

mean_error_RMS = sum(perViewErrors) / len(objpoints)

print('Número de imágenes detectadas: ', aptas)
print('focal length x:',K[0,0])
# print('focal length y:',K[1,1])
# print('optical centers (cx,cy)=(',K[0,2],',',K[1,2],')')
# print('distortion coefficients \n' ,'k1 = ',dist[0,0],'\n','k2 = ',dist[0,1],'\n','p1 = ',dist[0,2],'\n','p2 = ',dist[0,3],'\n','k3 = ',dist[0,4])
# print('translation vectors:',tvecs)
# print('rotation vectors:',rvecs)
# print('std intrinsics:',stdDeviationsIntrinsics)
# print('std extrinsics:',stdDeviationsExtrinsics)
# print('reprojection_errors:',perViewErrors)
# print('distortion coeffs',dist)
# print('mean reprojection_error:',mean_error_RMS)



# guardamos los datos

np.save('prueba_6x9/intrinsic_matrix',K)
np.save('prueba_6x9/distortion_coeffs',dist)
np.save('prueba_6x9/rotation_vecs',rvecs)
np.save('prueba_6x9/traslation_vecs',tvecs)
np.save('prueba_6x9/puntos_objeto',objpoints)
np.save('prueba_6x9/puntos_imagen',imgpoints)
np.save('prueba_6x9/errors',perViewErrors)
np.save('prueba_6x9/mean_error',mean_error_RMS)
