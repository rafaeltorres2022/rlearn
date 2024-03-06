import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from rlearn.cluster_utils import euclidian_distance

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