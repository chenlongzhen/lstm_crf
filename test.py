import numpy as np

a=np.arange(5)*3*10
a=np.asarray(a)
b=np.arange(3)*10
b=np.asarray(b)
tag=np.asarray([[1,6,1],[6,2,3],[7,1,2],[2,6,3],[7,9,1]])
print(b.reshape((1,-1)))
print(a.reshape((-1,1)))
print(a.reshape((-1,1))+b.reshape((1,-1))+tag)