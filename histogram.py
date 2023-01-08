import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# cargamos los datos a utilizar

error_x = np.load('data2/delta_x_list.npy')
error_y = np.load('data2/delta_y_list.npy')


# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(error_x, bins='auto', color='#0504aa',alpha=0.7,rwidth=0.95)

plt.grid(axis='y', alpha=0.75)
plt.xlabel('Valor del error de retroproyección')
plt.ylabel('Frequency')
plt.title('Histogram')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.style.use('ggplot')
plt.show()