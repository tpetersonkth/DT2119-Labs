import numpy as np
import matplotlib.pyplot as plt

m = np.load('saved_data.npy')
label = np.load('classification_label.npy')
result = np.load('classification_result.npy')
plt.pcolormesh(m.T)
plt.title('maximum likelihood')
plt.xlabel('utterances')
plt.ylabel('GMM Components')


plt.plot(label, label='Truth', linewidth=5, c='black')
plt.plot(result, label='Classification Result')
plt.legend()
plt.savefig('ML.png')
plt.show()
