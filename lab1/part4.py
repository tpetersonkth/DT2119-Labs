import numpy as np
from lab1 import proto
import matplotlib.pyplot as plt
from lab1.tools import *
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy
data = np.load('lab1_data.npz')['data']
example_data = np.load('lab1_example.npz')['example'].item()

target = 'lmfcc'
plt.figure(figsize=(10,20))
ax = plt.subplot(8, 1, 1)
ax.set_title('Samples')
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.plot(example_data['samples'], linewidth=0.5)

output = proto.mfcc(example_data['samples'])

plt.savefig('Results/part4.png', bbox_inches='tight')
plt.show()

