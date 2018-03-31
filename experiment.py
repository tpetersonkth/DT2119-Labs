import numpy as np

x = np.arange(24.).reshape(4,3,2)

x1 = np.square(x)
x2 = np.sum(x1, -1)
print(x2.shape)
out = np.tensordot(x,x, axes=(2,2))
print(out.shape)