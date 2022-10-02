import numpy as np
#print(np.zeros((3,3)))
#print(np.mgrid[0:3,0:5].T.reshape(-1,2))

a = np.array([[1,2,3], [4,5,6]])

#print(a.T) #traspuesto

#print(a.T.reshape(-1,2))

p = np.zeros((1, 2 * 5, 3), np.float32) #matrix de 10x3
p[0, :, :2] = np.mgrid[0:2, 0:5].T.reshape(-1, 2)


print(p)
