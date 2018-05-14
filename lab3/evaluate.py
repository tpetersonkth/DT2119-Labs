import numpy as np
import matplotlib.pyplot as plt
from StandardiseData import standardize_per_utterance, lmfcc_stack, standardize_per_training_set, get_training_and_validation_sets
from confusion_mat import plot_confusion_matrix, get_confusion_matrix
import pickle
import editdistance

NUM_STACK = 5

def combinePhonemes(labels,stateList):
    #labels - one dimensional tensor where each element is an observation
    spIndex = -1
    phoneme_list = []
    prev = ''
    for i, state in enumerate(stateList):
        if state == 'sp_0':
            spIndex = i
        c = state.split('_')[0]
        if c != prev:
            phoneme_list.append(c)
        prev = c

    labels[labels > spIndex] += 2
    labels = np.floor(labels/3)
    labels = labels.astype("int")

    return labels, phoneme_list

def groupedList(data):
    grouped = []
    prev = -1
    for item in data:
        if item != prev:
            grouped.append(item)
        prev = item
    return grouped


statlist = pickle.load(open('stateList.pkl', 'rb'))
p = np.load('predicted_test.npy')

train, validation = get_training_and_validation_sets(np.load('train_data.npz')['data'])
_, __, test_data = standardize_per_training_set(train, [], np.load('test_data.npz')['data'])

print('processing data for input')
x = [lmfcc_stack(d['lmfcc'], NUM_STACK) for d in test_data]
x = np.vstack(x)
y = [d['targets'] for d in test_data]
y = np.hstack(y).T


predicted = p.argmax(axis=1)
label = y.argmax(axis=1)
# combine into phenemse
predicted, phoneme_list = combinePhonemes(predicted, statlist)
label, _ = combinePhonemes(label, statlist)

grouped_label = groupedList(label)
grouped_predicted = groupedList(predicted)

accuracy = editdistance.eval(grouped_label, grouped_predicted) / len(grouped_label)
print('accuracy %0.2f%%' % accuracy)
cm = get_confusion_matrix(predicted, label)
# manually normalize sil_0 and sil_1
# cm[39, 39] = 0
# cm[40, 40] = 0
print('calculating accuracy')
accuracy = 1 - np.count_nonzero(predicted - label) / predicted.shape[0]
print(accuracy * 100, '%')
cm[13,13] = 0
plot_confusion_matrix(cm, phoneme_list, title='Phoneme state confusion mat, no silence',save=True)


# model = keras.models.load_model('h3_adagrad_relu_u256_e100.h5py')
# print('predicting..')
# p = model.predict(x)
print('hello world')