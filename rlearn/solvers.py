import numpy as np

class GradientDescent():

    def __init__(self, learning_rate, loss_function) -> None:
        self.lr = learning_rate
        self.loss_function = loss_function

    def step(self, ws, b, error, X):
        new_w = ws - self.loss_function.p_d_wrt_w(X, error, len(X)) * self.lr
        new_b = b - self.loss_function.p_d_wrt_b(error, len(X)) * self.lr
        return new_w, new_b

# class StochasticGradientDescent():

#     def __init__(self, learning_rate, loss_function, mini_batch) -> None:
#         self.lr = learning_rate
#         self.loss_function = loss_function
#         self.mini_batch = mini_batch

#     def step(self, theta, error, X):
#         x_batch = X.
#         new_w = theta - self.loss_function.p_d_wrt_w(x_batch, error, len(X)) * self.lr
#         new_b = theta - self.loss_function.p_d_wrt_b(error, len(X)) * self.lr
#         return new_w, new_b

def solver_selector(function_name, learning_rate, loss_function):
    stochastic = True
    if function_name == 'gd': return GradientDescent(
        learning_rate=learning_rate,
        loss_function=loss_function), ~stochastic
    elif function_name == 'sgd': return GradientDescent(
        learning_rate=learning_rate,
        loss_function=loss_function), stochastic
    else: raise ValueError(f'{function_name} is not a valid function.')
