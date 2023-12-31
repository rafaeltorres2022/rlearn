import numpy as np
import pandas as pd
from rlearn.activation_functions import activation_factory
from rlearn.metrics import loss_factory
from rlearn.solvers import solver_factory
from sklearn.metrics import mean_squared_error

class Perceptron():
    def __init__(self, learning_rate = 0.01, epochs = 1000, solver = 'sgd', 
                activation='linear', loss_function = 'mse', mini_batch=100, tol=0.001, patience = 10,
                verbose = 1) -> None:
        self.ws = []
        self.b = 0
        #self.lr = learning_rate
        self.epochs = epochs
        self.activation = activation_factory(activation)
        self.loss_function = loss_factory(loss_function)
        self.error = 0
        self.tol = tol
        self.verbose = verbose
        self.patience = patience
        self.mini_batch = mini_batch
        self.solver = solver_factory(solver, learning_rate, self.loss_function, mini_batch)
        self.history = {
            'bias' : [],
            'loss' : [],
            'error' : [],
        }

    def fit(self, X, y, verbose=1):
        X = self.check_if_df(X)
        y = self.check_if_df(y)
        self.initialize_wheights(X)
        for i in range(len(self.ws)):
            self.history[f'weights_{i}'] = []
                       
        for epoch in range(self.epochs):

            batch_X, batch_y = self.solver.define_batch(X, y)

            propagate = self.propagate(batch_X, self.ws, self.b)#(X * ws).sum(axis=1)+b
            
            error = batch_y - propagate
            
            #self.ws = self.ws - mse.p_d_wrt_w(X, error, len(y)) * self.lr
            #gradient_w_dir = 
            self.ws, self.b = self.solver.step(self.ws, self.b, error, batch_X, epoch)#.step(self.ws, self.loss_function.p_d_wrt_w(X, error, len(y)))

            #self.b = self.b - mse.p_d_wrt_b(error, len(y)) * self.lr 
            #gradient_b_dir = 
            #self.b = self.solver.step(self.b, self.loss_function.p_d_wrt_b(error, len(y)))

            #self.ws, self.b = self.step(self.ws, self.b, X, y)
            loss = self.loss_function.loss(y, self.propagate(X, self.ws, self.b))

            [self.history[f'weights_{i}'].append(self.ws[i]) for i in range(len(self.ws))]
            self.history['bias'].append(self.b)
            self.history['loss'].append(loss)
            self.history['error'].append(error)

            prev_loss = loss
            # print(f'prev_loss: {prev_loss} \n loss: {loss} \n diff: {prev_loss - loss}')
            # if (prev_loss - loss < self.loss_margin):
            #     break
            # else:
            #     prev_loss = loss
            #if (epoch % self.verbose == 0) & (self.verbose > 0):
            #    print(f'Epoch {epoch}: {(self.ws, self.b)} Loss: {loss} Lr: {self.solver.lr}')

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
        return self.propagate(X, self.ws, self.b)
    
    def propagate(self, X, ws, b):
        summation = (X.reshape(X.shape[0], -1) * ws).sum(axis=1)+b
        return self.activation(summation)
    
    def check_if_df(self, data):
        if (type(data) is pd.DataFrame):
            return data.to_numpy()
        elif (type(data) is pd.Series):
            return data.to_numpy()
        else:
            return data