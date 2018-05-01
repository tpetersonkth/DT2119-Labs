import lab3.lab3_tools as tools
<<<<<<< HEAD
import lab1.proto as lab1Proto
=======

tools.loadAudio("1a.wav")
>>>>>>> d2d51c390cfd61598ef9f5ab93bd3492ad3e6d0d

samples, samplingrateR = tools.loadAudio("1a.wav")
lmfcc = lab1Proto.mfcc(samples,samplingrate=samplingrateR)
print("Done")
