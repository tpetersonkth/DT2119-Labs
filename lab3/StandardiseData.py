import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical


def standardize_per_utterance(fname):
    a = np.load(fname)['data']
    #Fitting each utterance
    scaler = StandardScaler()
    standByUtterance = []
    for i in range(0,len(a)):
        newDatapoint = {}
        fittedScalar = scaler.fit(a[i]['lmfcc'])
        newDatapoint['lmfcc'] = fittedScalar.transform(a[i]['lmfcc'])
        newDatapoint['targets'] = one_hot(a[i]['targets'])
        standByUtterance.append(newDatapoint)
    return standByUtterance

def one_hot(target):
    return to_categorical(target,61).transpose()  #61 = the amount of possible states

stand = standardize_per_utterance("G:/train_data.npz")

print("Done")



