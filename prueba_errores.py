import numpy as np
import cv2 as cv
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.

images = glob.glob('Resources\*.jpg')
datos_errores = []
for num in np.arange(1,len(images)+1):
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    print(num)
    print(images[:num])
    for fname in images[:num]:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (7,6), corners2, ret)
            #cv.imshow('img', img)
            #cv.waitKey(0)
            # obtenemos la camera matrix, distortion coeffs, rotation and traslation vectors
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        else: print('error ', fname)

    # re-projection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
        total_error = mean_error / len(objpoints)
    print("total error of ", num, "photos:", total_error)

    datos_errores.append(total_error)

print(datos_errores)

cv.destroyAllWindows()

# REPRESENTACION DE LOS ERRORES

x = np.arange(1,len(images)+1)

plt.scatter(x, datos_errores)
plt.show()

