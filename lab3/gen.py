import numpy as np
import pickle
from lab3.lab3_proto import *
from lab1.proto import mfcc
from lab2.prondict import prondict
import os
from queue import Queue
from threading import Thread
SET = 'train' # test/train
THREADS = 20

phoneHMMs = np.load('lab3/lab2_models.npz')['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

def gen(fname):
    print(fname)
    samples, samplingrate = loadAudio(fname)
    lmfcc = mfcc(samples)

    wordTrans = list(path2info(fname)[2])
    phoneTrans = words2phones(wordTrans, prondict, addShortPause=True)
    hmms = concatHMMs(phoneHMMs, phoneTrans)
    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
                  for stateid in range(nstates[phone])]
    stateTrans_idx = list(map(stateList.index, stateTrans))
    aligned = forcedAlignment(lmfcc, hmms, stateTrans_idx)

    return aligned, lmfcc
threads = []
def threading(q:Queue, ret_q:Queue):
    data = []
    for i in range(THREADS):
        worker = Thread(target=work, args=(q,ret_q))
        threads.append(worker)
        worker.start()

    for t in threads:
        # Blocking
        t.join()

    while not ret_q.empty():
        fname, aligned, lmfcc = ret_q.get()
        data.append({'filename': fname, 'lmfcc': lmfcc, 'mspec': 'mspec', 'targets': aligned})

    np.savez('%s_data.npz' % SET, data=data)
    print('Done I think?')

def work(q:Queue, ret_q:Queue):
    while not q.empty():
        fname = q.get()
        aligned, lmfcc = gen(fname)
        ret_q.put((fname, aligned, lmfcc))
        q.task_done()



q = Queue()
ret_q = Queue()
i = 0
for root, dirs, files in os.walk('lab3/asset/tidigits/disc_4.1.1/tidigits/%s' % SET):
    for file in files:
        if file.endswith('.wav'):
            fname = os.path.join(root, file)
            q.put(fname)

threading(q, ret_q)


# aligned, lmfcc = gen(fname)
# data.append({'filename': fname, 'lmfcc': lmfcc,
#                   'mspec': 'mspec', 'targets': aligned})

