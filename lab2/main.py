import numpy as np
import matplotlib.pyplot as plt
import lab2.proto2 as proto2
import lab2.tools2 as tools2
import lab2.plotting as plotting
from lab2.prondict import prondict
from timeit import default_timer as timer

#Load Data
data = np.load('lab2_data.npz')['data']
example_data = np.load('lab2_example.npz')['example'].item()
lmfcc_example = example_data['lmfcc']
phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
lmfcc = data[10]['lmfcc']

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

#Temporary testing
#proto2.baum_welch(lmfcc_example, hmmTest['means'], hmmTest['covars'], log_startprob, log_trans, example_data)

loglikelihood = tools2.log_multivariate_normal_density_diag(lmfcc_example, hmmTest['means'], hmmTest['covars'])

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
log_gamma = proto2.statePosteriors(log_alpha,log_beta)
diffg = log_gamma - example_data['loggamma']

#Print execution time
print('Execution done in '+str(round((timer()-startTime),2))+" seconds")

#Plot graphs
#plt = plotting.plotEachStep(lmfcc_example,loglikelihood,log_alpha,log_beta,log_gamma)
#plt.show()

plotting.plotBestPath(log_alpha,viter[1])
plt.show()
