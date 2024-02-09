import numpy as np


class Solver:

    def __init__(self, learning_rate, mini_batch) -> None:
        self.lr = learning_rate
        self.mini_batch = mini_batch
    
    def step(self, ws, b, grad_w, grad_b):
        raise NotImplementedError()

    def define_batch(self, X, y):
        indexes = np.random.randint(0, high=X.shape[0], size=self.mini_batch)
        batch_x = X[indexes, :]
        batch_y = y[indexes,]
        return batch_x, batch_y
    
class PerceptronSolver(Solver):

    def __init__(self, learning_rate, mini_batch) -> None:
        super().__init__(learning_rate, mini_batch)

    def define_batch(self, X, y):
        return X, y
    
    def step(self, ws, b, grad_w, grad_b):
        return ws + grad_w, b + grad_b

class GradientDescent(Solver):

    def __init__(self, learning_rate, mini_batch, decay=0) -> None:
        super().__init__(learning_rate, mini_batch)
        self.decay = decay
        self.step_count = 1

    def step(self, ws, b, grad_w, grad_b):
        #self.lr = self.lr * np.e**(-(self.step_count*self.decay))
        self.step_count+=1
        new_ws = ws - grad_w * self.lr
        new_b = b - grad_b * self.lr
        #new_b = b - self.loss_function.p_d_wrt_b(error, len(X)) * self.lr
        return new_ws, new_b

    def define_batch(self, X, y):
        return X, y
    
class StochasticGradientDescent(Solver):

    def __init__(self, learning_rate, mini_batch, momentum=0.99) -> None:
        super().__init__(learning_rate, mini_batch)
        self.momentum = momentum
        self.vtw = 0
        self.vtb = 0

    def step(self, ws, b, grad_w, grad_b):
        self.vtw = self.momentum * self.vtw + self.lr * grad_w
        self.vtb = self.momentum * self.vtb + self.lr * grad_b
        new_ws = ws - self.vtw
        new_b = b - self.vtb
        return new_ws, new_b
    
class Adam(Solver):

    def __init__(self, mini_batch, learning_rate=0.001, beta1 = 0.9, beta2 = 0.999, ep = 10e-8) -> None:
        super().__init__(learning_rate, mini_batch)
        self.beta1 = beta1
        self.beta2 = beta2
        self.ep = ep
        self.mtw = 0
        self.mtb = 0
        self.vtw = 0
        self.vtb = 0
        self.step_count = 1

    def step(self, ws, b, grad_w, grad_b):
        self.mtw = (self.beta1 * self.mtw) + ((1 - self.beta1) * grad_w)
        self.mtb = (self.beta1 * self.mtb) + ((1 - self.beta1) * grad_b)
        self.vtw = self.beta2 * self.vtw + (1 - self.beta2) * grad_w**2
        self.vtb = self.beta2 * self.vtb + (1 - self.beta2) * grad_b**2
        m_hatw = self.mtw / (1 - self.beta1 ** self.step_count)
        m_hatb = self.mtb / (1 - self.beta1 ** self.step_count)
        v_hatw = self.vtw / (1 - self.beta2 ** self.step_count)
        v_hatb = self.vtb / (1 - self.beta2 ** self.step_count)
        new_ws = ws - self.lr * m_hatw / np.sqrt(v_hatw) + self.ep
        new_b = b - self.lr * m_hatb / np.sqrt(v_hatb) + self.ep
        self.step_count+=1
        return new_ws, new_b


class Adam2d():

    def __init__(self, layers_size, learning_rate=0.001, beta1 = 0.9, beta2 = 0.999, ep = 10e-8) -> None:
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.ep = ep
        self.mtw = [0 for _ in range(layers_size)]
        self.mtb = [0 for _ in range(layers_size)]
        self.vtw = [0 for _ in range(layers_size)]
        self.vtb = [0 for _ in range(layers_size)]
        self.epoch = 1

    def step(self, old_w, old_b, gradients_w, gradients_b, layer):
        self.mtw[layer] = (self.beta1 * self.mtw[layer]) + ((1 - self.beta1) * gradients_w)
        self.mtb[layer] = (self.beta1 * self.mtb[layer]) + ((1 - self.beta1) * gradients_b)
        self.vtw[layer] = self.beta2 * self.vtw[layer] + (1 - self.beta2) * gradients_w**2
        self.vtb[layer] = self.beta2 * self.vtb[layer] + (1 - self.beta2) * gradients_b**2
        m_hatw = self.mtw[layer] / (1 - self.beta1 ** self.epoch)
        m_hatb = self.mtb[layer] / (1 - self.beta1 ** self.epoch)
        v_hatw = self.vtw[layer] / (1 - self.beta2 ** self.epoch)
        v_hatb = self.vtb[layer] / (1 - self.beta2 ** self.epoch)
        self.epoch += 1
        new_w = old_w - self.lr * m_hatw / (np.sqrt(v_hatw) + self.ep)
        new_b = old_b - self.lr * m_hatb / (np.sqrt(v_hatb) + self.ep)
        return new_w, new_b
    

def solver_factory(function_name, learning_rate, mini_batch):
    if function_name == 'gd': return GradientDescent(
        learning_rate=learning_rate,
        mini_batch=mini_batch)
    elif function_name == 'sgd': return StochasticGradientDescent(
        learning_rate=learning_rate,
        mini_batch=mini_batch)
    elif function_name == 'adam': return Adam(
        learning_rate=learning_rate,
        mini_batch=mini_batch)
    elif function_name == 'perceptron': return PerceptronSolver(
        learning_rate=learning_rate,
        mini_batch=mini_batch)
    else: raise ValueError(f'{function_name} is not a valid function.')
