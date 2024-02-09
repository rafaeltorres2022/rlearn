import numpy as np
from numba import jit, njit

@njit
def backpooling(patches):
    '''
    Backpropagation through MaxPooling layer
    '''
    for item in range(patches.shape[0]):
        for row in range(patches.shape[1]):
            for col in range(patches.shape[2]):
                for channel in range(patches.shape[3]):
                    temp = patches[item, row, col, channel].flatten()
                    value = np.amax(temp)
                    index = np.argwhere(temp == value)[0][0]
                    temp[:] = 0
                    temp[index] = 1
                    patches[item, row, col, channel] = temp.reshape(patches.shape[-2], patches.shape[-1])
    return patches