import numpy as np
import pickle
from lab3.lab3_proto import *
from lab1.proto import mfcc
from lab2.prondict import prondict
import os

SET = 'train' # test/train

def gen(fname):
    phoneHMMs = np.load('lab3/lab2_models.npz')['phoneHMMs'].item()
    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
    with open('lab3/stateList.pkl', 'rb') as f:
        stateList = pickle.load(f)

    samples, samplingrate = loadAudio(fname)
    lmfcc = mfcc(samples)

    wordTrans = list(path2info(fname)[2])

    phoneTrans = words2phones(wordTrans,prondict, addShortPause=True)

    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
                      for stateid in range(nstates[phone])]



traindata = []
for root, dirs, files in os.walk('lab3/asset/tidigits/disc_4.1.1/tidigits/%s' % SET):
    for file in files:
        if file.endswith('.wav'):
            fname = os.path.join(root, file)
            print(fname)