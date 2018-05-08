import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical


def standardize_per_utterance(data):
    #Fitting each utterance
    scaler = StandardScaler()
    standByUtterance = []
    for i in range(0,len(data)):
        newDatapoint = {}
        fittedScalar = scaler.fit(data[i]['lmfcc'])
        newDatapoint['lmfcc'] = fittedScalar.transform(data[i]['lmfcc'])
        newDatapoint['targets'] = one_hot(data[i]['targets'])
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

def add_id_and_gender(data):
    for i in range(0,len(data)):
        splitted = data[i]['filename'].split("/")
        data[i]['gender'] = splitted[-3]
        data[i]['id'] = splitted[-2]
    return data

def get_data_by_speaker(data):
    dataBySpeaker = {}
    for i in range(0, len(data)):
        dataBySpeaker[data[i]['id']] = []

    for i in range(0, len(data)):
        dataBySpeaker[data[i]['id']].append(data[i])

    return dataBySpeaker


def standardize_per_speaker(data):
    data = add_id_and_gender(data)
    dataBySpeaker = get_data_by_speaker(data)

    data = []
    speakers = dataBySpeaker.keys()
    for speaker in speakers:
        fitData = [d['lmfcc'] for d in dataBySpeaker[speaker]]
        fitData = np.vstack(fitData)
        scaler = StandardScaler()
        fittedScalar = scaler.fit(fitData)
        for i in range(0,len(dataBySpeaker[speaker])):
            dataBySpeaker[speaker][i]['lmfcc'] = fittedScalar.transform(dataBySpeaker[speaker][i]['lmfcc'])
            data.append(dataBySpeaker[speaker][i])

    return data

def get_training_and_validation_sets(trainingData):
    men = []
    women = []
    a = len(trainingData)
    trainingData = add_id_and_gender(trainingData)
    trainingData = get_data_by_speaker(trainingData)
    keys = trainingData.keys()

    for speaker in keys:
        if trainingData[speaker][0]['gender'] == 'man':
            men.append(trainingData[speaker])
        else:
            women.append(trainingData[speaker])

    meni = int(0.1*len(men))
    womeni = int(0.1*len(women))
    validation = men[:meni] + women[:womeni]
    training = men[meni:] + women[womeni:]

    validationSet = []
    for i in validation:
        validationSet = validationSet + i

    trainingSet = []
    for i in training:
        trainingSet = trainingSet + i

    print("SPlit done")
    return trainingSet, validationSet

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
    data = get_training_and_validation_sets(trainingData)
    #data = standardize_per_speaker(trainingData)
    #data = standardize_per_training_set(trainingData)
    print("Done")