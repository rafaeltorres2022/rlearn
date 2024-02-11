import numpy as np

class Regularization():

    def __init__(self, alpha, l1_ratio) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def derivative(self, weights):
        ridge_wrt_w = self.alpha * (1 - self.l1_ratio) * weights
        lasso_wrt_w = self.alpha * self.l1_ratio * np.sign(weights)
        return ridge_wrt_w + lasso_wrt_w