import numpy as np

class MeanSquaredError:
    def __init__(self) -> None:
        pass
    
    def gradient(self, X, error, n):
        return np.append((-X.T * error).T.sum(axis=0) * 2/n, (error * -1).sum(axis=0) * 2/n)

    def loss(self, y, pred):
        summation = ((y - pred)**2).sum()
        return summation/len(y)
    
def loss_factory(function_name):
    if function_name == 'mse': return MeanSquaredError()
    else: raise ValueError(f'{function_name} is not a valid function.')