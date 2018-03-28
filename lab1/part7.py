import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

loaded = np.load('lab1_data.npz')
data = loaded['data']

y = data[0]['samples']
y2 = data[1]['samples']
size = 32
est = GaussianMixture(size)

stacked = np.vstack((y,y))

est.fit(stacked.T)

x = np.arange(0, len(y))
plt.scatter(y, y2, s=0.2)
plt.scatter(est.means_[:,0], [6] * size, color='red')
plt.show()

