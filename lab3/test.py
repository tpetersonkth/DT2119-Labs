import lab3.lab3_tools as tools
import lab3.lab3_proto as proto
import lab1.proto as lab1Proto
from lab2.prondict import *

tools.loadAudio("lab3/1a.wav")

samples, samplingrateR = tools.loadAudio("lab3/1a.wav")
lmfcc = lab1Proto.mfcc(samples,samplingrate=samplingrateR)

fname = 'lab3/asset/tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
samples, samplingrate = tools.loadAudio(fname)
lmfcc = lab1Proto.mfcc(samples)

wordlist = ['z','4','3']
proto.words2phones(wordlist,prondict)

path2infoOutput = tools.path2info(fname)[2]
wordTrans = list(tools.path2info(fname)[2])




print("Test Done")
