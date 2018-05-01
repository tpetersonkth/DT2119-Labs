import lab3.lab3_tools as tools
import lab1.proto as lab1Proto

tools.loadAudio("1a.wav")

samples, samplingrateR = tools.loadAudio("1a.wav")
lmfcc = lab1Proto.mfcc(samples,samplingrate=samplingrateR)
print("Done")
