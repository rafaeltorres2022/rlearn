import pytest
import numpy as np
from rlearn.activation_functions import *

def test_relu_activation():
    relu = Relu()

    values = [-10e-10, -100, -0.1, 0, 10e-10, 1, 10e10]
    results = []
    for value in values:
        results.append(relu.activate(value))

    assert results == [0, 0, 0, 0, 10e-10, 1, 10e10]

def test_relu_prime():
    relu = Relu()

    values = [-10e-10, -100, -0.1, 0, 10e-10, 1, 10e10]
    results = []
    for value in values:
        results.append(relu.prime(value))

    assert results == [0, 0, 0, 0, 1, 1, 1]

def test_leaky_relu_activation():
    leaky_relu = LeakyRelu()

    values = [-10e-10, -100, -0.1, 0, 10e-10, 1, 10e10]
    results = []
    for value in values:
        results.append(leaky_relu.activate(value))

    assert results == [-10e-10*0.001, -100*0.001, -0.1*0.001, 0, 10e-10, 1, 10e10]

def test_leaky_relu_prime():
    leaky_relu = LeakyRelu()

    values = [-10e-10, -100, -0.1, 0, 10e-10, 1, 10e10]
    results = []
    for value in values:
        results.append(leaky_relu.prime(value))

    assert results == [0.001, 0.001, 0.001, 0.001, 1, 1, 1]
