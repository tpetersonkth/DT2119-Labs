import numpy as np
from lab1 import proto
import matplotlib.pyplot as plt
loaded = np.load('lab1_data.npz')
data = loaded['data']

target = 'lmfcc'
giant_mfcc = []
for d in data:
    out = proto.mfcc(d['samples'])
    giant_mfcc.append(out)
    print(out.shape)

giant_mfcc = np.vstack(giant_mfcc)

# Answer for part 5
correlation = np.corrcoef(giant_mfcc)
# TODO last question for part 5

print(giant_mfcc)