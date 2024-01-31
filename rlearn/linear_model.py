import numpy as np
import pandas as pd
from rlearn.activation_functions import activation_factory
from rlearn.metrics import loss_factory
from rlearn.solvers import solver_factory
from rlearn.regularization import Regularization

class Perceptron():
    def __init__(self, learning_rate = 0.001, solver = 'sgd', 
                activation='linear', loss_function = 'mse', mini_batch=200) -> None:
        self.ws = None
        self.b = 0
        self.activation = activation_factory(activation)
        self.loss_function = loss_factory(loss_function)
        self.mini_batch = mini_batch
        self.solver = solver_factory(solver, learning_rate, mini_batch)
        self.history = []

    def fit(self, X, y, epochs = 1000, verbose=5):
        self.initialize_wheights(X)
                       
        for epoch in range(epochs):

            batch_X, batch_y = self.solver.define_batch(X, y)

            propagate = self.propagate(batch_X)
            grad_w, grad_b = self.loss_function.gradient(batch_X, batch_y - propagate)
            self.ws, self.b = self.solver.step(self.ws, self.b, grad_w, grad_b)#.step(self.ws, self.loss_function.p_d_wrt_w(X, error, len(y)))
            loss = self.loss_function.loss(y, self.propagate(X))

            self.history.append(loss)


        return self.ws, self.b

    def initialize_wheights(self, X):
        self.ws = np.random.normal(0, 1, X.shape[1])

    def predict(self, X):
        pred = self.propagate(X)
        if self.activation.__class__.__name__ == 'Sigmoid':
            return (pred > 0.5).astype('int8')
        return pred
    
    def propagate(self, X):
        summation = (X.dot(self.ws))+self.b
        return self.activation.activate(summation)

class ElasticNet:

    def __init__(self, alpha = 0.1, l1_ratio = 0.5, learning_rate=0.1, solver='gd', loss='mse') -> None:
        self.weights = None
        self.bias = 0
        self.regularization = Regularization(alpha=alpha, l1_ratio=l1_ratio)
        self.hist = []
        self.solver = solver_factory(solver, learning_rate, 200)
        self.loss_function = loss_factory(loss)


    def fit(self, X, y, epochs=10000, verbose=5):
        self.weights = np.random.normal(0, 1, size=X.shape[1])

        for epoch in range(epochs):
            foward = X.dot(self.weights) + self.bias
            grad_w, grad_b = self.loss_function.gradient(X, y-foward)
            
            loss = self.loss_function.loss(y, foward)
            self.hist.append(loss)
            
            grad_w += self.regularization.derivative(self.weights)
            self.weights, self.bias = self.solver.step(self.weights, self.bias, grad_w, grad_b)

            if epoch % (epochs/verbose) == 0:
                print(self.hist[-1])

    def predict(self, X):
        return X.dot(self.weights) + self.bias
            