import numpy as np

def euclidian_distance(point1, point2):
        return np.sqrt(((point1 - point2) ** 2).sum())