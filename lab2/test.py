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

#Calculate loglikelihood(log_emslik)
loglikelihood = tools2.log_multivariate_normal_density_diag(lmfcc, hmmTest['means'], hmmTest['covars'])
#diff = example_data['obsloglik'] - loglikelihood

#Forward alogithm

log_alpha = proto2.forward(loglikelihood, log_startprob ,log_trans)
diffa = log_alpha - example_data['logalpha']

#Viterbi alogithm
viter = proto2.viterbi(loglikelihood, log_startprob ,log_trans)
diffv1 = viter[0] - example_data['vloglik'][0]
diffv2 = viter[1] - example_data['vloglik'][1]


#Backward alogithm
ref = example_data['logbeta']
log_beta = proto2.backward(loglikelihood, log_startprob ,log_trans)
#diff = log_beta - ref



#Print execution time
print('Execution done in '+str(round((timer()-startTime),2))+" seconds")


