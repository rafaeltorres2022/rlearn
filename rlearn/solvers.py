import numpy as np


class Solver:

    def __init__(self, learning_rate, loss_function, mini_batch) -> None:
        self.lr = learning_rate
        self.loss_function = loss_function
        self.mini_batch = mini_batch
    
    def step(self, ws, b, error, X, epoch):
        raise NotImplementedError()

    def define_batch(self, X, y):
        indexes = np.random.randint(0, high=X.shape[0], size=self.mini_batch)
        try:
            batch_x = X[indexes, :]
            batch_y = y[indexes,]
        except IndexError:
            batch_x = X[indexes]
            batch_y = y[indexes]
        return batch_x, batch_y
    
class GradientDescent(Solver):

    def __init__(self, learning_rate, loss_function, mini_batch) -> None:
        super().__init__(learning_rate, loss_function, mini_batch)

    def step(self, ws, b, error, X, epoch):
        gradient = self.loss_function.gradient(X, error)
        new_ws = np.append(ws, b) - gradient * self.lr
        #new_b = b - self.loss_function.p_d_wrt_b(error, len(X)) * self.lr
        return new_ws[:-1], new_ws[-1]

    def define_batch(self, X, y):
        return X, y
    
class StochasticGradientDescent(Solver):

    def __init__(self, learning_rate, loss_function, mini_batch) -> None:
        super().__init__(learning_rate, loss_function, mini_batch)

    def step(self, ws, b, error, X, epoch):
        gradient = gradient = self.loss_function.gradient(X, error)
        new_ws = np.append(ws, b) - gradient * self.lr
        return new_ws[:-1], new_ws[-1]
    
class Adam(Solver):

    def __init__(self, loss_function, mini_batch, learning_rate=0.001, beta1 = 0.9, beta2 = 0.999, ep = 10e-8) -> None:
        super().__init__(learning_rate, loss_function, mini_batch)
        self.beta1 = beta1
        self.beta2 = beta2
        self.ep = ep
        self.mt = 0
        self.vt = 0

    def step(self, ws, b, error, X, epoch):
        epoch += 1
        gradient = gradient = self.loss_function.gradient(X, error)
        self.mt = (self.beta1 * self.mt) + ((1 - self.beta1) * gradient)
        self.vt = self.beta2 * self.vt + (1 - self.beta2) * gradient**2
        m_hat = self.mt / (1 - self.beta1 ** epoch)
        v_hat = self.vt / (1 - self.beta2 ** epoch)
        new_ws = np.append(ws, b) - self.lr * m_hat / np.sqrt(v_hat) + self.ep
        ws, b = new_ws[:-1], new_ws[-1]
        return ws, b

def solver_factory(function_name, learning_rate, loss_function, mini_batch):
    if function_name == 'gd': return GradientDescent(
        learning_rate=learning_rate,
        loss_function=loss_function,
        mini_batch=mini_batch)
    elif function_name == 'sgd': return StochasticGradientDescent(
        learning_rate=learning_rate,
        loss_function=loss_function,
        mini_batch=mini_batch)
    elif function_name == 'adam': return Adam(
        learning_rate=learning_rate,
        loss_function=loss_function,
        mini_batch=mini_batch)
    else: raise ValueError(f'{function_name} is not a valid function.')
