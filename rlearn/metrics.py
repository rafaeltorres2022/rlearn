import numpy as np

class MeanSquaredError:
    def __init__(self) -> None:
        pass
    
    def p_d_wrt_w(self, X, error, n):
        return (-X.T * error).T.sum(axis=0) * 2/n

    def p_d_wrt_b(self, error, n):
        return (error * -1).sum(axis=0) * 2/n

    def calculate(self, y, pred):
        summation = ((y - pred)**2).sum()
        return summation/len(y)
    
def loss_selector(function_name):
    if function_name == 'mse': return MeanSquaredError()
    else: raise ValueError(f'{function_name} is not a valid function.')