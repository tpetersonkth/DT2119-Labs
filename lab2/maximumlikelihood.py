import numpy as np
import matplotlib.pyplot as plt
import lab2.proto2 as proto2
import lab2.tools2 as tools2
from lab2.prondict import prondict
from timeit import default_timer as timer

#Load Data
data = np.load('lab2_data.npz')['data']
phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()

# Create modellist from prondict
modellist = {}
for digit in prondict.keys():
    modellist[digit] = ['sil'] + prondict[digit] + ['sil']

loglik = np.zeros((len(data),len(modellist)))
ground_truth = []
classification = []
for i, utterance in enumerate(data):
    print('utterance', i)

    # Getting true result
    digit = utterance['digit']
    if digit == 'o': truth = 0
    elif digit == 'z': truth = 1
    else: truth = int(digit) + 1
    ground_truth.append(truth)

    for j, modelkey in enumerate(modellist.keys()):
        lmfcc = utterance['lmfcc']

        #Generate HMM for the word zero
        hmmTest = proto2.concatHMMs(phoneHMMs,modellist[modelkey])
        log_startprob = np.log(hmmTest['startprob'])
        log_trans = np.log(hmmTest['transmat'])[:-1, :-1]

        loglikelihood = tools2.log_multivariate_normal_density_diag(lmfcc, hmmTest['means'], hmmTest['covars'])

        #Forward alogithm
        log_alpha = proto2.forward(loglikelihood, log_startprob ,log_trans)
        loglik[i,j] = tools2.logsumexp(log_alpha[-1])

    classified = np.argmax(loglik[i,:])
    classification.append(classified)

print(classification)
print(ground_truth)

np.save('saved_data', loglik)

print('hello')

