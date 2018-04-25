import numpy as np
import matplotlib.pyplot as plt

m = np.load('saved_data.npy')
plt.pcolormesh(m)
plt.title('maximum likelihood')
plt.ylabel('utterances')
plt.xlabel('GMM Likelihood')
plt.savefig('ML.png')
plt.show()
