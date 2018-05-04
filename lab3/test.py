import numpy as np

nn = np.tile(np.arange(1,10),(2,1)).T
print(nn)

def stack(matrix:np.ndarray, n):
    stacked = []
    # backward:
    for i in np.arange(-n, 0):
        new_mat = np.zeros(matrix.shape)
        new_mat[-i:] = matrix[:i]
        view = matrix[1:-i+1, :]
        new_mat[:-i] = view[::-1, :]
        stacked.append(new_mat)

    stacked.append(matrix)

    # forward:
    for i in np.arange(1, n+1):
        new_mat = np.zeros(matrix.shape)
        new_mat[:-i] = matrix[i:]
        view = matrix[-i-1:-1,:]
        new_mat[-i:] = view[::-1,:]
        stacked.append(new_mat)
    ndstacked = np.hstack(stacked)
    return ndstacked


stack(nn, 5)