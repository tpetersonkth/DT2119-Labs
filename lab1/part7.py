import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from lab1 import proto

loaded = np.load('lab1_data.npz')
data = loaded['data']
size = 32
target = 'lmfcc'
giant_mfcc = []
for d in data:
    out = proto.mfcc(d['samples'])
    giant_mfcc.append(out)


giant_mfcc = np.vstack(giant_mfcc)

est = GaussianMixture(size)

est.fit(giant_mfcc)

x = np.arange(0, len(y))
plt.scatter(y, s=0.2)
plt.scatter(est.means_[:,0], [6] * size, color='red')
plt.show()

