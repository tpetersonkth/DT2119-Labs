import numpy as np
from lab3.lab3_tools import *

from lab2.proto2 import *

def words2phones(wordList, pronDict, addSilence=True, addShortPause=False):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    phones = []
    if addSilence:
        phones = ['sil']


    for word in wordList:
        phones += pronDict[word]
        if addShortPause:
            phones += ['sp']

    if addSilence:
        phones += ['sil']

    return phones

def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of states in each HMM model (could be different for each)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    wordHmm = copy.deepcopy(hmmmodels[namelist[0]])
    states = namelist.count('sp') + (len(namelist) - namelist.count('sp'))*3
    d = 4*len(namelist)-(len(namelist)-1)
    wordHmm['transmat'] = np.zeros((states+1,states+1))
    wordHmm['means'] = np.zeros((states,13))
    wordHmm['covars'] = np.zeros((states, 13))
    wordHmm['startprob'] = np.zeros((1, states))
    wordHmm['startprob'][0,0] = 1
    c = 0
    for i in range(0,len(namelist)):
        if (namelist[i] == 'sp'):
            wordHmm['transmat'][c:c + 2, c:c + 2] = hmmmodels[namelist[i]]['transmat']
            wordHmm['means'][c:c + 1, :] = hmmmodels[namelist[i]]['means']
            wordHmm['covars'][c:c + 1, :] = hmmmodels[namelist[i]]['covars']
            c += 1
        else:
            wordHmm['transmat'][c:c+4, c:c+4] = hmmmodels[namelist[i]]['transmat']
            wordHmm['means'][c:c+3,:] = hmmmodels[namelist[i]]['means']
            wordHmm['covars'][c:c + 3, :] = hmmmodels[namelist[i]]['covars']
            c += 3
    wordHmm['transmat'][-1, -1] = 1
    return wordHmm

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """
    log_emlik = log_multivariate_normal_density_diag(lmfcc, phoneHMMs['means'], phoneHMMs['covars'])
    states = viterbi(log_emlik,phoneHMMs['startprob'],phoneHMMs['transmat'])
    #TODO max pooling?
    aligned = phoneTrans[states]
    return aligned




def hmmLoop(hmmmodels, namelist=None):
    """ Combines HMM models in a loop

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to combine, if None,
                 all the models in hmmmodels are used

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models
       stateMap: map between states in combinedhmm and states in the
                 input models.

    Examples:
       phoneLoop = hmmLoop(phoneHMMs)
       wordLoop = hmmLoop(wordHMMs, ['o', 'z', '1', '2', '3'])
    """
