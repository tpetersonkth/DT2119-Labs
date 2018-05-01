import lab3.lab3_proto
import numpy as np
import pickle

from lab3.lab3_tools import *
from lab1.proto import mfcc
#Get stateList
'''
phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

#with open('stateList.pkl', 'wb') as f:
#    pickle.dump(stateList, f)
'''

with open('lab3/stateList.pkl', 'rb') as f:
    stateList = pickle.load(f)

fname = 'lab3/asset/tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
samples, samplingrate = loadAudio(fname)
lmfcc = mfcc(samples)


print("Done")