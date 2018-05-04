import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense

NUM_HIDDEN_LAYERS = 4 # number of hidden layers
HIDDEN_U = 256

data = np.load('lab3/train_data.npz')['data']

x = [d['lmfcc'] for d in data]
x = np.vstack(x)

model = Sequential()
model.add(Dense(HIDDEN_U,activation='relu', input_shape=(5,)))

for h in range(NUM_HIDDEN_LAYERS-1):
    model.add(Dense(HIDDEN_U,activation='relu'))

model.add(Dense(HIDDEN_U,activation='softmax'))
