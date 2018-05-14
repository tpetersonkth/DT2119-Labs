import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Activation, Dense
import matplotlib.pyplot as plt
from StandardiseData import standardize_per_utterance, lmfcc_stack, standardize_per_training_set, get_training_and_validation_sets

NUM_STACK = 5

train, validation = get_training_and_validation_sets(np.load('train_data.npz')['data'])
_, __, test_data = standardize_per_training_set(train, [], np.load('test_data.npz')['data'])

print('processing data for input')
x = [lmfcc_stack(d['lmfcc'], NUM_STACK) for d in test_data]
x = np.vstack(x)
y = [d['targets'] for d in test_data]
y = np.vstack(y)
p = np.load('predicted_test.npz')

#model = keras.models.load_model('h3_adagrad_relu_u256_e100.h5py')
print('predicting..')
#p = model.predict(x)
print('hello world')