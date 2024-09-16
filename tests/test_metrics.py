import pytest
import numpy as np
from rlearn.metrics import *

def test_mse_loss():
    y_hat = np.array([43, 44, 45, 46, 47])
    y = np.array([41, 45, 49, 47, 44])
    mse = MeanSquaredError()

    assert mse.loss(y_hat, y) == 6.2