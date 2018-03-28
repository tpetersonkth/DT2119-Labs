# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, hamming
from lab1.tools import *
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
import scipy
def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)

    mspec = logMelSpectrum(spec, samplingrate)
    ceps = cepstrum(mspec, nceps)
    return lifter(ceps, liftercoeff)


# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """

    start = 0
    frames = []
    while True:
        end = start + winlen
        if end > len(samples):
            break;
            #end=-1

        frame = samples[start:end]
        frames.append(frame)
        start += winshift

    npframes = np.vstack(frames)
    #plt.pcolormesh(npframes.T)
    #plt.show()
    return npframes

    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    emphasized = lfilter([1, -p], 1, input)
    return emphasized


def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    window = hamming(input.shape[1], sym=0)
    npwindows = np.multiply(input, window)
    return npwindows

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    freqDom = fft(input,nfft)
    absVal = np.absolute(freqDom)
    retVal = np.square(absVal)
    return retVal

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in tools.py to calculate the filterbank shapes and
          nmelfilters
    """

    bank = trfbank(samplingrate, input.shape[1])
    ret = input.dot(bank.T)
    ret = np.log(ret)
    return ret


def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """

    out = dct(input, norm='ortho')
    out = out[:, 0:13]
    #out = lifter(out)
    return out

def calcDist(x,y):
    dist = (x-y)
    dist = np.sqrt(dist.dot(dist.T))
    return dist

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the  code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    N = x.shape[0] #Columns
    M = y.shape[0] #Rows
    accD = [[0 for x in range(M+1)] for y in range(N+1)]
    for i in range(1, N+1):
        accD[i][0] = 9999999999
    for i in range(1, M+1):
        accD[0][i] = 9999999999
    accD[0][0] = 0



    for i in range(1,N+1):
        for j in range(1,M+1):
            accD[i][j] = dist(x[i-1], y[j-1]) + min(accD[i-1][j], accD[i-1][j-1], accD[i][j-1])

    return accD[N][M]/(N+M)

def compareUtterances(data):
    D = [[0 for x in range(44)] for y in range(44)]

    mfccMatrix = [0 for x in range(44)]

    for i in range(0, 44):
        mfccMatrix[i] = mfcc(data[i]['samples'])

    for i in range(0,44):
        p = (i/44)*100
        print("%f%"%P)

        for j in range(0, 44):
            D1 = mfccMatrix[i]
            D2 = mfccMatrix[j]
            D[i][j] = dtw(D1,D2,calcDist)

    np.save('insurance d.txt', D)
    plt.pcolormesh(D)
    #plt.savefig()
    plt.show()
    return D

