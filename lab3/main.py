import lab3.lab3_proto
import numpy as np
import pickle

#Get stateList
'''
phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

#with open('stateList.pkl', 'wb') as f:
#    pickle.dump(stateList, f)
'''

with open('stateList.pkl', 'rb') as f:
    stateList = pickle.load(f)

print("Done")