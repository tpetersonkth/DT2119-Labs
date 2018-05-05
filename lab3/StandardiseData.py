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

def standardize_per_training_set(trainingData,validationData,testData):
    #trainingData = np.load(trainingFname)['data']
    #testData = np.load(testFname)['data']

    fitData = [d['lmfcc'] for d in trainingData]
    fitData = np.vstack(fitData)
    scaler = StandardScaler()
    fittedScalar = scaler.fit(fitData)

    trainingSet = []
    for i in range(0,len(trainingData)):
        newDatapoint = {}
        newDatapoint['lmfcc'] = fittedScalar.transform(trainingData[i]['lmfcc'])
        newDatapoint['targets'] = one_hot(trainingData[i]['targets'])
        trainingSet.append(newDatapoint)

    validationSet = []
    for i in range(0, len(validationData)):
        newDatapoint = {}
        newDatapoint['lmfcc'] = fittedScalar.transform(validationData[i]['lmfcc'])
        newDatapoint['targets'] = one_hot(validationData[i]['targets'])
        validationSet.append(newDatapoint)

    testSet = []
    for i in range(0, len(testData)):
        newDatapoint = {}
        newDatapoint['lmfcc'] = fittedScalar.transform(testData[i]['lmfcc'])
        newDatapoint['targets'] = one_hot(testData[i]['targets'])
        testSet.append(newDatapoint)

    return trainingSet,validationSet ,testSet



def lmfcc_stack(matrix:np.ndarray, n):
    stacked = []
    # backward:
    for i in np.arange(-n, 0):
        new_mat = np.zeros(matrix.shape)
        new_mat[-i:] = matrix[:i]
        view = matrix[1:-i+1, :]
        new_mat[:-i] = view[::-1, :]
        stacked.append(new_mat)

    stacked.append(matrix)

    # forward:
    for i in np.arange(1, n+1):
        new_mat = np.zeros(matrix.shape)
        new_mat[:-i] = matrix[i:]
        view = matrix[-i-1:-1,:]
        new_mat[-i:] = view[::-1,:]
        stacked.append(new_mat)
    ndstacked = np.hstack(stacked)
    return ndstacked


if __name__ == "__main__":
    trainingData = np.load("G:/train_data.npz")['data']
    data = standardize_per_training_set(trainingData)
    print("Done")