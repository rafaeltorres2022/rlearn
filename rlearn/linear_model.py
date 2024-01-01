import numpy as np
import pandas as pd
from rlearn.activation_functions import activation_factory
from rlearn.metrics import loss_factory
from rlearn.solvers import solver_factory
from sklearn.metrics import mean_squared_error

class Perceptron():
    def __init__(self, learning_rate = 0.0001, epochs = 1000, solver = 'sgd', 
                activation='linear', loss_function = 'mse', mini_batch=100, tol=0.001, patience = 10,) -> None:
        self.ws = []
        self.b = 0
        #self.lr = learning_rate
        self.epochs = epochs
        self.activation = activation_factory(activation)
        self.loss_function = loss_factory(loss_function)
        self.error = 0
        self.tol = tol
        self.patience = patience
        self.mini_batch = mini_batch
        self.solver = solver_factory(solver, learning_rate, self.loss_function, mini_batch)
        self.history = {
            'bias' : [],
            'loss' : [],
        }

    def fit(self, X, y, save_weights_history = False):
        X = self.check_if_df(X)
        y = self.check_if_df(y, is_target=True)
        self.initialize_wheights(X)
        if save_weights_history == True:
            for i in range(len(self.ws)):
                self.history[f'weights_{i}'] = []
                       
        for epoch in range(self.epochs):

            batch_X, batch_y = self.solver.define_batch(X, y)

            propagate = self.propagate(batch_X, self.ws, self.b)#(X * ws).sum(axis=1)+b
            error = batch_y - propagate
            self.ws, self.b = self.solver.step(self.ws, self.b, error, batch_X, epoch)#.step(self.ws, self.loss_function.p_d_wrt_w(X, error, len(y)))
            loss = self.loss_function.loss(y, self.propagate(X, self.ws, self.b))

            if save_weights_history == True:
                [self.history[f'weights_{i}'].append(self.ws[i]) for i in range(len(self.ws))]
                self.history['bias'].append(self.b)
            self.history['loss'].append(loss)


        return self.ws, self.b
    
    def initialize_loss(self, X, y, ws, b, metric):
        pred = self.propagate(X, ws, b)
        return metric.calculate(y, pred)

    def initialize_wheights(self, X):
        try:
            self.ws = np.ones(X.shape[1])
        except IndexError as e:
            self.ws = np.ones(1)

    def predict(self, X):
        X = self.check_if_df(X)
        pred = self.propagate(X, self.ws, self.b)
        if self.activation.__name__ == 'sigmoid':
            binarize = lambda x : 1 if (x > 0.5) else 0
            binarize = np.vectorize(binarize)
            return binarize(pred)
        return pred
    
    def propagate(self, X, ws, b):
        summation = (X * ws).sum(axis=1)+b
        return self.activation(summation)
    
    def check_if_df(self, data, is_target=False):
        if (type(data) is pd.DataFrame):
            return data.to_numpy()
        elif (type(data) is pd.Series) and (is_target == False):
            return data.to_numpy().reshape(data.shape[0], -1)
        elif (type(data) is pd.Series) and is_target:
            return data.to_numpy()
        elif (type(data) == np.ndarray) and (data.shape == (data.shape[0], ) and (is_target==False)):
            return data.reshape(-1, 1)
        else:
            return data