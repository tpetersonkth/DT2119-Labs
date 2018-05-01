import lab3.lab3_proto
import numpy as np
import pickle

from lab3.lab3_tools import *
from lab3.lab3_proto import *
import lab2.proto2 as proto2
from lab1.proto import mfcc
from lab2.prondict import prondict
#Get stateList
'''


#with open('stateList.pkl', 'wb') as f:
#    pickle.dump(stateList, f)
'''

phoneHMMs = np.load('lab3/lab2_models.npz')['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
with open('lab3/stateList.pkl', 'rb') as f:
    stateList = pickle.load(f)



fname = 'lab3/asset/tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
samples, samplingrate = loadAudio(fname)
lmfcc = mfcc(samples)

wordTrans = list(path2info(fname)[2])
phoneTrans = words2phones(wordTrans,prondict, addShortPause=True)
hmms = concatHMMs(phoneHMMs,phoneTrans)
stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
                  for stateid in range(nstates[phone])]

aligned = forcedAlignment(lmfcc, hmms, stateTrans)
print("Done")