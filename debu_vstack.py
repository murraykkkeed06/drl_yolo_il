import numpy as np

a = [1,0,1,0,1,0,1]

b = [3,4,3,4,3,4,3]

a = np.vstack((a,b))

print(a.shape)
print(type(b))