import numpy as np
from scipy.special import expit

def relu(x) -> float:
    return np.maximum(0, x)

def linear(x) -> float:
    return x

def sigmoid(x):
    return expit(x)

def activation_factory(function_name):
    if function_name == 'relu': return relu
    elif function_name == 'linear': return linear
    elif function_name == 'sigmoid': return sigmoid
    else: raise ValueError(f'{function_name} is not a valid function.')