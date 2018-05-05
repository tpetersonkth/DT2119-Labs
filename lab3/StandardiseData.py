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

def standardize_per_training_set(trainingFname,testFname):
    trainingData = np.load(trainingFname)['data']
    testData = np.load(testFname)['data']

    fitData = [d['lmfcc'] for d in trainingData]
    fitData = np.vstack(fitData)
    scaler = StandardScaler()
    fittedScalar = scaler.fit(fitData)

    standByTrainingSet = []
    for i in range(0,len(trainingData)):
        newDatapoint = {}
        newDatapoint['lmfcc'] = fittedScalar.transform(trainingData[i]['lmfcc'])
        newDatapoint['targets'] = one_hot(trainingData[i]['targets'])
        standByTrainingSet.append(newDatapoint)

    standByTestSet = []
    for i in range(0, len(testData)):
        newDatapoint = {}
        newDatapoint['lmfcc'] = fittedScalar.transform(testData[i]['lmfcc'])
        newDatapoint['targets'] = one_hot(testData[i]['targets'])
        standByTestSet.append(newDatapoint)

    return standByTrainingSet, standByTestSet

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
    data = standardize_per_training_set("G:/train_data.npz","G:/test_data.npz")
    print("Done")