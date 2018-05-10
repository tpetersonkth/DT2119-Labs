import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
import matplotlib.pyplot as plt
from StandardiseData import standardize_per_utterance, lmfcc_stack, get_training_and_validation_sets
import sys

NUM_STACK = 5
INPUT_DIM = 13 * (NUM_STACK * 2 + 1)

# Hyper params
NUM_HIDDEN_LAYERS = 3 # number of hidden layers
ACTIVATION = sys.argv[1]
OPTIMIZER = sys.argv[2]
HIDDEN_U = sys.argv[3]
BATCH_SIZE = sys.argv[4]
EPOCH = sys.argv[5]
# with open('train_data.npz') as file:
#     data = file['data']

print("Activation function:"+str(sys.argv[1]))
print("Optimizer:"+str(sys.argv[2]))
print("Hidden_U:"+str(sys.argv[3]))
print("Batch_size:"+str(sys.argv[4]))
print("Epoch:"+str(sys.argv[5]))

train, validation = get_training_and_validation_sets(np.load('train_data.npz')['data'])
# Standarrdize dataset
train = standardize_per_utterance(train)
validation = standardize_per_utterance(validation)

x = [lmfcc_stack(d['lmfcc'], NUM_STACK) for d in train]
x = np.vstack(x)

y = [d['targets'] for d in train]
y = np.hstack(y).T

# validation set
x_val = [lmfcc_stack(d['lmfcc'], NUM_STACK) for d in validation]
x_val = np.vstack(x_val)
y_val = [d['targets'] for d in validation]
y_val = np.hstack(y_val).T

input_dim = x.shape[1]
output_dim = y.shape[1]
model = Sequential()
model.add(Dense(HIDDEN_U,activation=ACTIVATION, input_shape=(input_dim,)))

for h in range(NUM_HIDDEN_LAYERS-1):
    model.add(Dense(HIDDEN_U,activation=ACTIVATION))

model.add(Dense(output_dim,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER)

history = model.fit(x, y, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCH)

fname = 'h%d_%s_%s_u%d_e%d.h5py' % (NUM_HIDDEN_LAYERS, OPTIMIZER, ACTIVATION,HIDDEN_U,EPOCH)
model.save(fname)
print('I"m done!!')

