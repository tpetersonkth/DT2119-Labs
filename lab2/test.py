import numpy as np
from lab2 import proto2
import matplotlib.pyplot as plt
import lab2.tools2
from lab2.prondict import prondict
from lab2.tools2 import *
data = np.load('lab2_data.npz')['data']
example_data = np.load('lab2_example.npz')['example'].item()

phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()

modellist = {}
for digit in prondict.keys():
    modellist[digit] = ['sil'] + prondict[digit] + ['sil']

hmmTest = proto2.concatHMMs(phoneHMMs,modellist['o'])
ex = example_data['lmfcc']
loglikelihood = log_multivariate_normal_density_diag(ex, hmmTest['means'], hmmTest['covars'])
diff = example_data['obsloglik'] - loglikelihood
print('done')


