import numpy as np
from lab1 import proto
import matplotlib.pyplot as plt
from lab1.tools import *
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy
data = np.load('lab1_data.npz')['data']
example_data = np.load('lab1_example.npz')['example'].item()
''' 
target = 'lmfcc'
#plt.plot(example_data['samples'], linewidth=0.5)
output = proto.mfcc(example_data['samples'])
if np.array_equal(output, example_data[target]):
    print('it works!')

plt.subplot(2, 1, 1)
plt.title('Spec')
plt.pcolormesh(example_data[target].T)
plt.subplot(2, 1, 2)
plt.pcolormesh(output.T)
plt.show()


diff = example_data[target] - output
'''
D = proto.compareUtterances(data)

D = np.load('insurance d.txt.npy')


Z = linkage(D, method='complete')
labels = tidigit2labels(data)

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
p = dendrogram(Z, labels=labels)
plt.show()
print('done')

#distances = proto.calcDist()
#output =
#plt.pcolormesh(output.T)
#plt.show()

