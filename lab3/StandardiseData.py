import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



def standardize_per_utterance(fname):
    a = np.load(fname)['data']
    #Fitting each utterance
    standByUtterance = []
    for i in range(0,len(a)):
        newDatapoint = {}
        fittedScalar = scaler.fit(a[i]['lmfcc'])
        newDatapoint['lmfcc'] = scaler.transform(a[i]['lmfcc'])
        newDatapoint['targets'] = a[i]['targets'].copy()
        standByUtterance.append(newDatapoint)
    return standByUtterance

print("Done")



