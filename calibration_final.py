import numpy as np
import cv2 as cv
import glob

#dimension del chessboard
chessboard = (6,7)

#termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#preparamos los puntos objeto
objp = np.zeros((chessboard[0]*chessboard[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard[1],0:chessboard[0]].T.reshape(-1,2)

# Arrays para almacenar los puntos objeto y los puntos imagen
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

#array con los paths de las imagenes que se van a utilizar
images = glob.glob('Resources\*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,6), None) #esta funcion encuentra las esquinas del chessboard
    if ret == True:  #si el pattern ha sido detectado
        objpoints.append(objp) #añade puntos objeto
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria) #esta funcion refina el colocamiento de las esquinas
        imgpoints.append(corners2) #añade los puntos imagen refinados
        cv.drawChessboardCorners(img, (7,6), corners2, ret) #dibuja los puntos en el chessboard
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()

#obtenemos la camera matrix, distortion coeffs, rotation and traslation vectors
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print('focal length x:',mtx[0,0])
print('focal length y:',mtx[1,1])
print('optical centers (cx,cy)=(',mtx[0,2],',',mtx[1,2],')')
print('distortion coefficients:',dist)
print('translation vectors:',tvecs)
print('rotation vectors:',rvecs)
#re-projection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )