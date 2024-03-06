import numpy as np
from scipy.stats import mode
from rlearn.cluster_utils import euclidian_distance

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
            result.append(self.define_result(self.y[np.argsort([euclidian_distance(point1, point2) for point2 in self.X])[1:self.k+1]]))
        return result
    
    def define_result(self, args):
        return mode(args).mode

class KNearestNeighborsRegressor(KNearestNeighbors):

    def define_result(self, args):

        return np.mean(args)