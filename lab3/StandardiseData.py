import numpy as np
from sklearn.preprocessing import StandardScaler

def standardize_per_utterance(fname):
    a = np.load(fname)['data']
    #Fitting each utterance
    scaler = StandardScaler()
    standByUtterance = []
    for i in range(0,len(a)):
        newDatapoint = {}
        fittedScalar = scaler.fit(a[i]['lmfcc'])
        newDatapoint['lmfcc'] = fittedScalar.transform(a[i]['lmfcc'])
        newDatapoint['targets'] = a[i]['targets'].copy()
        standByUtterance.append(newDatapoint)
    return standByUtterance

def one_hot(target, maxNum):
    pass

stand = standardize_per_utterance("G:/train_data.npz")

print("Done")



