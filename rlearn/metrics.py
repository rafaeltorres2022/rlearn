import numpy as np

class MeanSquaredError:
    def __init__(self) -> None:
        pass
    
    def gradient(self, X, error):
        return (-X.T * error).T.sum(axis=0) * 2/len(X), (error * -1).sum(axis=0) * 2/len(X)

    def loss(self, y, pred):
        summation = ((y - pred)**2).sum()
        return summation/len(y)
    
class LogLoss:
    
    def gradient(self, X, error):
        error = error * -1
        #return np.append(1/len(X) * X.T.dot(error), 1/len(X) * error.sum())
        return (X.T * error).T.sum(axis=0) * 1/len(X), (error).sum(axis=0) * 1/len(X)

    def loss(self, y, pred, ep = 10e-8):
        return np.average(-(y * np.log(pred + ep) + (1 - y) * np.log(1 - pred + ep)))
    
class PerceptronLoss:

    def gradient(self, X, error):
        return error.dot(X), error.sum()

    def loss(self, y, pred):
        return (y-pred).sum()

def loss_factory(function_name):
    if function_name == 'mse': return MeanSquaredError()
    elif function_name == 'logloss': return LogLoss()
    elif function_name == 'perceptron': return PerceptronLoss()
    else: raise ValueError(f'{function_name} is not a valid function.')