import numpy as np
from lab1 import proto
import matplotlib.pyplot as plt
data = np.load('lab1_data.npz')['data']
example_data = np.load('lab1_example.npz')['example'].item()

#plt.plot(example_data['samples'], linewidth=0.5)
output = proto.mfcc(example_data['samples'])
if np.array_equal(output, example_data['windowed']):
    print('it works!')



plt.subplot(2, 1, 1)
plt.title('Spec')
plt.pcolormesh(example_data['spec'].T)
plt.subplot(2, 1, 2)
plt.pcolormesh(output.T)
plt.show()

diff =  example_data['spec'] - output
print(data)