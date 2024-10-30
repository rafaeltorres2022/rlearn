import numpy as np
from scipy.stats import norm

class GaussianNB():

    def __init__(self):
        self.dists = []
        self.priors = []
        self.classes = []

    def fit(self, X, y):
        cols = X.shape[1]
        self.classes = np.unique(y)

        for col in range(cols):
            self.dists.append([])
            for clas in self.classes:
                rows_from_clas = X[np.argwhere(y==clas).flatten(), col]
                self.dists[col].append(
                    norm(
                        np.mean(rows_from_clas),
                        np.std(rows_from_clas)))
                
        for clas in np.unique(y):
            self.priors.append(len(np.argwhere(y == clas)) / len(y))

    def predict_row(self, row):
        right_class = 0
        best_score = 0
        for clas, prior in enumerate(self.priors):
            score = np.log(prior)
            for col in range(row.shape[0]):
                score+=np.log(self.dists[col][clas].pdf(row[col]))
            if (score >= best_score) or (clas == 0):
                best_score = score
                right_class = self.classes[clas]

        return right_class
        
    def predict(self, X):
        return np.apply_along_axis(self.predict_row, 1, X)