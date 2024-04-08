import numpy as np
from scipy.stats import mode
from rlearn.cluster_utils import euclidian_distance
from rlearn.tree import Node

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
    

class KDTree:

    def __init__(self, max_depth, min_leaf_size = 10) -> None:
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
    
    def fit(self, X):
        self.k = X.shape[1]
        self.X = X
        self.root = self.build_node(np.array([a for a in range(len(self.X))]), 0)
        

    def build_node(self, indexes, depth):
        node = Node()


        axis = depth % self.k

        median = np.median(self.X[indexes, axis])
        node.col_split = axis
        node.value_to_split = median
        
        slice_ids = np.where(self.X[indexes,axis] <= median)[0]
        others = np.where(~(self.X[indexes,axis] <= median))[0]
        
        if (depth == self.max_depth) | (len(slice_ids) < self.min_leaf_size) | (len(others) < self.min_leaf_size):
            node.set_data(indexes)
            node.output = 0
            node.is_leaf = True
            return node
        
        node.left_child = self.build_node(indexes[slice_ids], depth+1)
        node.right_child = self.build_node(indexes[others], depth+1)
        return node
    
    def predict(self, data, return_node=False):
        return np.apply_along_axis(self.predict_row, axis=1, arr=data)

    def predict_row(self, row):
        next_node = self.root
        for _ in range(self.max_depth):
            next_node = next_node.left_child if (row[next_node.col_split] <= next_node.value_to_split) else next_node.right_child
            if next_node.is_leaf:
                return next_node
            
    def region_query(self, X, index, eps=0.5, return_distances=False):
        neighbours = self.predict(X[index:index+1])[0].data
        temp = []
        dists = []
        for index_ in neighbours:
            distance = euclidian_distance(X[index], X[index_])
            if distance <= eps:
                dists.append(distance)
                temp.append(index_)
        
        sorted_indexes = np.argsort(dists)
        temp = np.array(temp)[sorted_indexes]
        dists = np.array(dists)[sorted_indexes]
        if return_distances:
            return dists, temp
        return temp