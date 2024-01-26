import numpy as np
from scipy.special import expit, softmax as soft

class Relu:

    def activate(self, x) -> float:
        return np.maximum(0, x)
    
    def prime(self, x):
        return 1*(x>0)

class LeakyRelu:

    def __init__(self, a=0.001) -> None:
        self.a = a
    
    def activate(self, x) -> float:
        return np.where(x>0, x, x*self.a)
    
    def prime(self, x):
        return 1*(x>0) + self.a*(x<0)

class Linear:
    
    def activate(self, x) -> float:
        return x
    
    def prime(self, x):
        return 1

class Sigmoid:

    def activate(self, x):
        return expit(x)

class Softmax:

    def activate(self, x):
        return soft(x, axis=1)

def activation_factory(function_name):
    if function_name == 'relu': return Relu()
    elif function_name == 'leakyrelu': return LeakyRelu()
    elif function_name == 'linear': return Linear()
    elif function_name == 'sigmoid': return Sigmoid()
    elif function_name == 'softmax': return Softmax()
    else: raise ValueError(f'{function_name} is not a valid function.')