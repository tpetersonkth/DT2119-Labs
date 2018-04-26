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
lmfcc4Ma = data[10]['lmfcc']
lmfcc4Mb = data[11]['lmfcc']
lmfcc4Fa = data[32]['lmfcc']
lmfcc4Fb = data[33]['lmfcc']
lmfcc6Ma = data[14]['lmfcc']
lmfcc6Mb = data[15]['lmfcc']
lmfcc6Fa = data[36]['lmfcc']
lmfcc6Fb = data[37]['lmfcc']

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

#plotting.plotBestPath(log_alpha,viter[1])
#plt.show()

#Plot 4.1
''' 
hmmTest = proto2.concatHMMs(phoneHMMs,modellist['6'])
Prob6Ma = tools2.log_multivariate_normal_density_diag(lmfcc6Ma, hmmTest['means'], hmmTest['covars'])
Prob6Mb = tools2.log_multivariate_normal_density_diag(lmfcc6Mb, hmmTest['means'], hmmTest['covars'])
Prob6Fa = tools2.log_multivariate_normal_density_diag(lmfcc6Fa, hmmTest['means'], hmmTest['covars'])
Prob6Fb = tools2.log_multivariate_normal_density_diag(lmfcc6Fa, hmmTest['means'], hmmTest['covars'])

plt.figure(figsize=(10, 10))
ax = plt.subplot(4, 1, 1)
plt.yticks(np.arange(0, 18 + 1, 1.0))
ax.set_title('Prob6Ma')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(6)
ax.set_xticklabels([])
plt.pcolormesh(np.transpose(Prob6Ma))

ax = plt.subplot(4, 1, 2)
plt.yticks(np.arange(0, 18 + 1, 1.0))
ax.set_title('Prob6Mb')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(6)
ax.set_xticklabels([])
plt.pcolormesh(np.transpose(Prob6Mb))

ax = plt.subplot(4, 1, 3)
plt.yticks(np.arange(0, 18 + 1, 1.0))
ax.set_title('Prob6Fa')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(6)
ax.set_xticklabels([])
plt.pcolormesh(np.transpose(Prob6Fa))

ax = plt.subplot(4, 1, 4)
plt.yticks(np.arange(0, 18 + 1, 1.0))
ax.set_title('Prob6Fb')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(6)
ax.set_xticklabels([])
plt.pcolormesh(np.transpose(Prob6Fb))

#plt.show()
'''

#Get best scores for all utterances
viterbiTable = np.zeros((44,11))
#modellistKeys = list(modellist.keys())
modellistKeys = ['z','o', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ylabels = []
for i in range(0,44):
    lmfcc = data[i]['lmfcc']
    normalize = len(data[0]['lmfcc'])
    ylabels.append(data[i]['digit'] + data[i]['gender'] + data[i]['repetition'])
    print(str(i))
    for j in range(0,11):
        hmm = proto2.concatHMMs(phoneHMMs, modellist[modellistKeys[j]])
        loglikelihood = tools2.log_multivariate_normal_density_diag(lmfcc, hmm['means'], hmm['covars'])
        log_startprob = np.log(hmm['startprob'])
        log_trans = np.log(hmm['transmat'])[:-1, :-1]
        viter = proto2.viterbi(loglikelihood, log_startprob ,log_trans)
        viterbiTable[i,j] = viter[0]/normalize


plt.figure(figsize=(10,10))
ax = plt.subplot(1, 1, 1)
plt.yticks(np.arange(0, 45, 1.0))
ax.set_title('Viterbi Table')
ax.set_yticklabels(ylabels)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(6)
plt.xticks(np.arange(0, 11, 1.0))
ax.set_xticklabels(modellistKeys)
plt.pcolormesh(viterbiTable)
plt.show()

print("Done")
