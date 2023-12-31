import numpy as np

def relu(x) -> float:
    return np.maximum(0, x)

def linear(x) -> float:
    return x

def activation_factory(function_name):
    if function_name == 'relu': return relu
    elif function_name == 'linear': return linear
    else: raise ValueError(f'{function_name} is not a valid function.')