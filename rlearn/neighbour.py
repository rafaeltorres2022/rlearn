import numpy as np
from scipy.stats import mode

class KNearestNeighbors:

    def __init__(self, k) -> None:
        self.k = k
        self.X = None
        self.y = None


    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()

    def predict(self, X):
        result = []
        for point1 in X:
            result.append(self.define_result(y_[np.argsort([euclidian_distance(point1, point2) for point2 in X_])[1:self.k+1]]))
        return result
    
    def define_result(self, args):
        return mode(args).mode

class KNearestNeighborsRegressor(KNearestNeighbors):

    def define_result(self, args):

        return np.mean(args)