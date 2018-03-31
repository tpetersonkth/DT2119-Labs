import numpy as np
from lab1 import proto
import matplotlib.pyplot as plt
loaded = np.load('lab1_data.npz')
data = loaded['data']

target = 'lmfcc'
giant_mfcc = []
giant_mspec = []
for d in data:
    out, mspec = proto.mfcc(d['samples'], liftering=False)
    giant_mfcc.append(out)
    giant_mspec.append(mspec)


giant_mfcc = np.vstack(giant_mfcc)
giant_mspec = np.vstack(giant_mspec)

# Answer for part 5
correlation_mfcc = np.corrcoef(giant_mfcc)
correlation_mspec = np.corrcoef(giant_mspec)


plt.figure(figsize=(20,10))
ax = plt.subplot(1, 2, 1)
ax.set_title('MFCC')
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.pcolormesh(correlation_mfcc)

ax = plt.subplot(1, 2, 2)
ax.set_title('MSPEC')
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.pcolormesh(correlation_mspec)

plt.show()

print(giant_mfcc)