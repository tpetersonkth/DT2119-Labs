import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
import matplotlib.pyplot as plt
from lab3.StandardiseData import standardize_per_utterance, lmfcc_stack

NUM_HIDDEN_LAYERS = 4 # number of hidden layers
HIDDEN_U = 256
BATCH_SIZE = 256
NUM_STACK = 5
INPUT_DIM = 13 * (NUM_STACK * 2 + 1)

# data = standardize_per_utterance('lab3/train_data.npz')
# x = [lmfcc_stack(d['lmfcc'], NUM_STACK) for d in data]
# x = np.vstack(x)
# y = [d['targets'] for d in data]
# y = np.hstack(y).T
# np.save('x', x)
# np.save('y',y)
x, y = np.load('x.npy'), np.load('y.npy')
input_dim = x.shape[1]
output_dim = y.shape[1]
model = Sequential()
model.add(Dense(HIDDEN_U,activation='relu', input_shape=(input_dim,)))

for h in range(NUM_HIDDEN_LAYERS-1):
    model.add(Dense(HIDDEN_U,activation='relu'))

model.add(Dense(output_dim,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd')

history = model.fit(x, y, validation_split=0.1, batch_size=BATCH_SIZE, epochs=1)
model.save('done_training')
print('I"m done!!')

