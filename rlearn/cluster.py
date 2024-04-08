import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from rlearn.cluster_utils import euclidian_distance
from rlearn.neighbour import KDTree
import seaborn as sns

class KMeans:

    def __init__(self, k) -> None:
        self.centroids = None
        self.k = k
        self.history = []
        self.centroids_history = []
    
    def initialize_centroids(self, X):
        return X[np.random.choice(range(len(X)), size=self.k)]

    def choose_closest_centroid(self, point, centroids=None):
        if centroids is not None:
            return np.argmin([euclidian_distance(point, centroid) for centroid in centroids])
        else:
            return np.argmin([euclidian_distance(point, centroid) for centroid in self.centroids])

    def choose_new_centroids(self, X, new_labels):
        for k_ in range(len(self.centroids)):
            self.centroids[k_] = np.apply_along_axis(np.mean, axis=0, arr = X[new_labels == k_])

    def get_inertia(self, X, new_labels):
        total_inertia = 0
        for k_, centroid in enumerate(self.centroids):
            total_inertia += np.apply_along_axis(euclidian_distance, axis=1, arr=X[new_labels == k_], point2 = centroid).sum()
        return total_inertia

    def fit(self, X, tol = 0.01):
        self.centroids = self.initialize_centroids(X)
        self.centroids_history.append(self.centroids.copy())
        for _ in range(10):
            new_labels = np.apply_along_axis(self.choose_closest_centroid, axis=1, arr = X)
            self.choose_new_centroids(X, new_labels)
            self.history.append(self.get_inertia(X, new_labels))
            self.centroids_history.append(self.centroids.copy())
            if self.should_it_stop(tol):
                break
            
    def should_it_stop(self, tol):
        try:
            return (self.history[-2] - self.history[-1]) <= tol
        except:
            return False

    def predict(self, X):
        return np.apply_along_axis(self.choose_closest_centroid, axis=1, arr = X)

    def make_animation(self, X, filename='kmeans.gif', interval = 500, features=(0,1)):
        fig = plt.figure()
        points = plt.scatter(x=X[:,features[0]], y=X[:,features[1]])
        cs = plt.scatter(x=[], y=[], c='red', label='Centroids')
        plt.legend()
        def update(frame):
            fig.suptitle(f'Epoch: {frame}')
            labels = np.apply_along_axis(self.choose_closest_centroid, axis=1, arr = X, centroids=self.centroids_history[frame])
            points.set_array(labels)
            cs.set_offsets(np.c_[self.centroids_history[frame][:,features[0]], self.centroids_history[frame][:,features[1]]])
            return cs, points
        ani = FuncAnimation(fig = fig, func=update, frames = len(self.centroids_history), interval=interval)
        ani.save(filename=filename, writer='pillow')
        plt.close()

class DBSCAN:

    def __init__(self, algorithm = 'brute', dist_func = euclidian_distance, eps = 0.5, min_points = 5, max_depth= 'auto', min_leaf_size = 10) -> None:
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.dist_func = dist_func
        self.eps = eps
        self.min_points = min_points
        self.labels = []
        self.kdtree = None
        self.algorithm_name = algorithm
        self.history = []

    def fit(self, X):
        self.labels = np.zeros(len(X))
        cluster_id = 1

        if self.algorithm_name == 'kdtree':
            if self.max_depth == 'auto':
                self.max_depth = X.shape[1] -1
            self.kdtree = KDTree(self.max_depth, self.min_leaf_size)
            self.kdtree.fit(X)
        self.algorithm = self.pick_algo(self.algorithm_name)

        for index in range(len(X)):
            if self.labels[index] == 0:
                if self.expand_cluster(X, index, cluster_id):
                    cluster_id+=1

    def expand_cluster(self, X, index, cluster_id):
        seeds = self.algorithm(X, index, self.eps)
        if len(seeds) < self.min_points:
            self.labels[index] = -1 # Noise
            self.history.append((index, -1))
            return False
        else:
            self.labels[seeds] = cluster_id
            [self.history.append((id_, cluster_id)) for id_ in seeds]
            seeds = np.delete(seeds, np.argwhere(seeds == index))

            while len(seeds) > 0:
                current_point = seeds[0]
                result = self.algorithm(X, current_point, self.eps)
                if len(result) >= self.min_points:
                    for r in result:
                        if (self.labels[r] == 0) | (self.labels[r] == -1):
                            if (self.labels[r] == 0):
                                seeds = np.append(seeds, r)
                            self.labels[r] = cluster_id
                            self.history.append((r, cluster_id))
                
                seeds = np.delete(seeds, np.argwhere(seeds == current_point))
            return True

    def pick_algo(self, algo):
        if algo == 'brute':
            return self.region_query
        elif algo == 'kdtree':
            return self.kdtree.region_query
        
    def region_query(self, X, index, eps):
        temp = []
        for index_ in range(len(X)):
            if euclidian_distance(X[index], X[index_]) < eps:
                temp.append(index_)
        return temp
    
    def make_animation(self, X, filename='dbscan.gif', interval = 50, features=(0,1)):
        fig = plt.figure()
        labels_anim = np.zeros(len(self.labels))
        print('it may take a while...')
        def init():
            points = sns.scatterplot(x=X[:,0], y=X[:,1])
            return points

        def update(frame):
            fig.clf()
            fig.suptitle(f'Epoch: {frame}')
            history_tuple = self.history[frame]
            labels_anim[history_tuple[0]] = history_tuple[1]
            points = sns.scatterplot(x=X[:,features[0]], y=X[:,features[1]], hue = labels_anim)
            points.get_legend().remove()
            return fig, points
        ani = FuncAnimation(fig = fig, func=update, init_func = init, frames = len(self.history), interval=interval, cache_frame_data=False)
        ani.save(filename=filename, writer='pillow')
        print('Done!')
        plt.close()