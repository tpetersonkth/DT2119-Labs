import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

a = np.load('G:/train_data.npz')['data']


#Fitting each utterance
standByUtterance = []
for i in range(0,len(a)):
    newDatapoint = {}
    fittedScalar = scaler.fit(a[i]['lmfcc'])
    newDatapoint['lmfcc'] = scaler.transform(a[i]['lmfcc'])
    newDatapoint['targets'] = a[i]['targets'].copy()
    standByUtterance.append(newDatapoint)

print("Done")



