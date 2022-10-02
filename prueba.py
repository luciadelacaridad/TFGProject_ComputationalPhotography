import cv2 as cv
import sys


img = cv.imread(r'C:\Users\USER\PycharmProjects\TFGProject\starry_night.png') #esta función carga la imagen

if img is None:
    sys.exit('Could not read the image')  #para ver si la imagen se carga correctamente

cv.imshow('Starry Night',img) #función para mostrar la imagen
k = cv.waitKey(0) #para que la imagen se deje de mostar cuando pulsamos cualquier tecla

if k == ord('s'):
    cv.imwrite('starry_night.png',img) #la imagen se guarda en la carpeta actual si pulsamos s

