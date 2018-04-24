import numpy as np
import matplotlib.pyplot as plt
import lab2.proto2 as proto2
import lab2.tools2 as tools2
from lab2.prondict import prondict
from timeit import default_timer as timer

#Load Data
data = np.load('lab2_data.npz')['data']
example_data = np.load('lab2_example.npz')['example'].item()
lmfcc = example_data['lmfcc']
phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()

#Start timer
startTime = timer()

#Create modellist from prondict
modellist = {}
for digit in prondict.keys():
    modellist[digit] = ['sil'] + prondict[digit] + ['sil']

#Generate HMM for the word zero
hmmTest = proto2.concatHMMs(phoneHMMs,modellist['o'])
log_startprob = np.log(hmmTest['startprob'])
log_trans = np.log(hmmTest['transmat'])[:-1, :-1]


proto2.baum_welch(lmfcc,hmmTest['means'], hmmTest['covars'], log_startprob, log_trans, example_data)

#Forward alogithm
log_alpha = proto2.forward(loglikelihood, log_startprob ,log_trans)
diffa = log_alpha - example_data['logalpha']

#Viterbi alogithm
viter = proto2.viterbi(loglikelihood, log_startprob ,log_trans)
diffv1 = viter[0] - example_data['vloglik'][0]
diffv2 = viter[1] - example_data['vloglik'][1]


#Backward alogithm
log_beta = proto2.backward(loglikelihood, log_startprob ,log_trans)
diffb = log_beta - example_data['logbeta']

#State Posteriors
log_gamma = proto2.statePosteriors(example_data['logalpha'],example_data['logbeta'])
diffg = log_gamma - example_data['loggamma']

log_gamma = proto2.updateMeanAndVar(lmfcc,example_data['loggamma'])
diffg = log_gamma - ref


#Print execution time
print('Execution done in '+str(round((timer()-startTime),2))+" seconds")

#Plot graphs
plt.figure(figsize=(20,10))
ax = plt.subplot(1, 1, 1)
ax.set_title('lmfcc')
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.pcolormesh(np.transpose(lmfcc))
plt.show()

plt.figure(figsize=(20,10))
ax = plt.subplot(1, 1, 1)
plt.title('lmfcc')
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.pcolormesh(np.transpose(lmfcc))
plt.show()

plt.figure(figsize=(20,10))
ax = plt.subplot(1, 1, 1)
ax.set_title('log alpha')
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.pcolormesh(np.transpose(log_alpha))
plt.show()


