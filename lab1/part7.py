import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from lab1 import proto

loaded = np.load('lab1_data.npz')
data = loaded['data']
size = 12
target = 'lmfcc'
giant_mfcc = []

seven = []
unrelated = []
i = 0
for d in data:
    out = proto.mfcc(d['samples'])
    giant_mfcc.append(out)

    #voice seven
    if i in [16, 17, 38, 39]:
        seven.append(out)
    if i in [5,35, 20]:
        unrelated.append(out)
    i += 1


giant_mfcc = np.vstack(giant_mfcc)

est = GaussianMixture(size)

est.fit(giant_mfcc)

plt.figure()
probs = []
pred = []

for i, utt in enumerate(seven):
    prob = est.predict_proba(utt)
    pred.append(est.predict(utt))
    probs.append(prob)
    # if i < 2:
    #     plt.plot(prob, color='blue')
    # else:
    #     plt.plot(prob, color='green')
for utt in unrelated:
    prob = est.predict_proba(utt)
    pred.append(est.predict(utt))
    probs.append(prob)
    # plt.plot(prob, color='red')



# plt.figure()
# for i in range(len(probs)):
#     #plt.subplot(3, 2, i+1)
#     plt.plot(probs[i])
plt.show()
plt.figure()
plt.plot(pred[4], label='Male 5 (reference)', color='lightgray')
plt.plot(pred[5], label='Female 4 (reference)', color='lightgray', linestyle='--')
plt.plot(pred[6], label='Female 1 (reference)', color='lightgray', linestyle=':')

plt.plot(pred[0], label='Male 7 (A)')
plt.plot(pred[1], label='Male 7 (B)')
plt.plot(pred[2], label='Female 7 (A)')
plt.plot(pred[3], label='Female 7 (B)')

plt.title('Posterior %d-GMM component for utterances' % size)

#plt.plot(pred[4], label='Female 5 (A)')
plt.legend()
plt.savefig('Results/GMM_posterior_line.png', pad_inches=0, bbox_inches='tight' )
plt.show()
# GMM Posterior single


# GMM Posterior
fig = plt.figure()
fig.suptitle('Posterior of each utterances vs gaussian means')

ax = plt.subplot(2, 2, 1)
ax.set_axis_off()
ax.set_title('Male pronouncing 7 (A)')
plt.pcolormesh(probs[0])

ax = plt.subplot(2, 2, 2)
ax.set_axis_off()
ax.set_title('Male pronouncing 7 (B)')
plt.pcolormesh(probs[1])

ax = plt.subplot(2, 2, 3)
ax.set_axis_off()
ax.set_title('Female pronouncing 7')
plt.pcolormesh(probs[2])

ax = plt.subplot(2, 2, 4)
ax.set_axis_off()
ax.set_title('Male pronouncing 5')
plt.pcolormesh(probs[3])
plt.savefig('Results/GMM_Posterior.png', pad_inches=0, bbox_inches='tight')
plt.show()


#Correlation
fig = plt.figure()
fig.suptitle('Correlation between utterance and GMM posterior')
corr1 = np.corrcoef(probs[0], probs[1])
corr2 = np.corrcoef(probs[0], probs[2])
corr3 = np.corrcoef(probs[0], probs[4])
ax = plt.subplot(2, 2, 1)
ax.set_title('Male 7 (A) vs Male 7 (B)')
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
plt.pcolormesh(corr1)

ax = plt.subplot(2, 2, 2)
ax.set_axis_off()
ax.set_title('Male 7 vs Female 7')
plt.pcolormesh(corr2)

ax = plt.subplot(2, 2, 3)
ax.set_axis_off()
ax.set_title('Male 7 vs Male 5')
plt.pcolormesh(corr3)

plt.savefig('Results/GMM_correlation.png', pad_inches=0, bbox_inches='tight')
plt.show()



