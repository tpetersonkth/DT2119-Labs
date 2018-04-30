import lab2.proto2 as proto2
import lab2.tools2 as tools2
from lab2.prondict import prondict
import numpy as np

N = 15 # number of observations
D = 2 # muber of dimensions
M = 3 # number of states

# define extreme posteriors:
gamma = np.zeros((N, M))
gamma[:4,0] = 1.0 # first 4 observations belong to state 0
gamma[4:10,1] = 1.0 # next 6 observations belong to state 1
gamma[10:,2] = 1.0 # the remaining observations belong to state 2

# generate some data
np.random.seed(100)
X = np.random.rand(N,D)

means, covars = proto2.updateMeanAndVar(X, np.log(gamma), varianceFloor=0.0)

# check differences with standard mean and var:
d0 = means[0,:] - np.mean(X[:4],  axis=0)
d1 = means[1,:] - np.mean(X[4:10],axis=0)
d2 = means[2,:] - np.mean(X[10:], axis=0)
d3 = covars[0,:] - np.var(X[:4],  axis=0)
d4 = covars[1,:] - np.var(X[4:10],axis=0)
d5 = covars[2,:] - np.var(X[10:], axis=0)