# este programa dibuja un cubo con una esquina en el primer punto detectado en el tablero
#
import numpy as np
import cv2 as cv
import glob

# cargamos los datos
mtx = np.load('data/intrinsic_matrix.npy')
dist = np.load('data/distortion_coeffs.npy')

# dimension del chessboard
chessboard = (9,6)

# definimos una función que dibuje el cubo
def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # dibuja el suelo
    img = cv.drawContours(img, [imgpts[:4]],-1,(255,153,153),-3)

    # dibuja las columnas
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255,153,204),3)

    # dibuja la tapa
    img = cv.drawContours(img, [imgpts[4:]],-1,(255,153,204),3)
    return img


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # termination criteria

objp = np.zeros((chessboard[0]*chessboard[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard[1],0:chessboard[0]].T.reshape(-1,2) # puntos de la forma (0,0,0),(1,0,0),(2,0,0),etc

# ejes del cubo
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0], [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

numero_imagen = 0
for fname in glob.glob('imagenes/*.png'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (chessboard[1],chessboard[0]),None)
    if ret == True:
        numero_imagen += 1
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # vectores de rotación y traslación
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        #
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,imgpts)
        cv.imshow('img',img)
        cv.imwrite('pose_estimation_cubo/cubo'+str(numero_imagen)+'.png',img)
        cv.waitKey()
cv.destroyAllWindows()

