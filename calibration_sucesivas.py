import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# dimension del chessboard
chessboard = (9,6)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# preparamos los puntos objeto
objp = np.zeros((chessboard[0]*chessboard[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard[1],0:chessboard[0]].T.reshape(-1,2) # puntos de la forma (0,0,0),(1,0,0),(2,0,0),etc


# array con los paths de las imágenes que se van a utilizar

images = glob.glob('imagenes/*.png')

x = []
err = []
f_x = []
f_y = []
c_x = []
c_y = []
k1 = []
k2 = []
k3 = []
p1 = []
p2 = []



for j in range(len(images)):
    aptas = 0  # contador para ver cuantos patterns se detectan
    imgpoints = []
    objpoints = []
    for fname in images[0:j+1]:        # bucle para cada imagen
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
    cv.destroyAllWindows()

    # obtenemos la camera matrix, distortion coeffs, rotation and traslation vectors

    ret, mtx, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors= cv.calibrateCameraExtended(objpoints, imgpoints, gray.shape[::-1], None, None)
    error = sum(perViewErrors)/len(objpoints)
    # print('numero de imagenes:',aptas)
    # print(mtx,dist,error)


    #vectores para representar
    x.append(aptas)
    err.append(error)
    f_x.append(mtx[0,0])
    f_y.append(mtx[1,1])
    c_x.append(mtx[0,2])
    c_y.append(mtx[1,2])
    k1.append(dist[0,0])
    k2.append(dist[0,1])
    p1.append(dist[0,2])
    p2.append(dist[0,3])
    k3.append(dist[0,4])


# fig, ([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2)
plt.scatter(x,err,label='error de retroproyeccion')
# ax.xaxis.set_major_locator(FixedLocator(np.arange(1,len(x)+1)))

# ax1.scatter(x,f_x,label='f_x')
# ax1.scatter(x,f_y,label='f_y')
# ax2.scatter(x,c_x,label='c_x')
# ax2.scatter(x,c_y,label='c_y')
# ax3.scatter(x,k1,label='k1')
# ax3.scatter(x,k2,label='k2')
# ax3.scatter(x,k3,label='k3')
# ax4.scatter(x,p1,label='p1')
# ax4.scatter(x,p2,label='p2')
#
# ax1.xaxis.set_major_locator(FixedLocator(np.arange(1,len(x)+1)))
# ax1.legend()
# ax2.xaxis.set_major_locator(FixedLocator(np.arange(1,len(x)+1)))
# ax2.legend()
# ax3.xaxis.set_major_locator(FixedLocator(np.arange(1,len(x)+1)))
# ax3.legend()
# ax4.xaxis.set_major_locator(FixedLocator(np.arange(1,len(x)+1)))
# ax4.legend()
plt.show()