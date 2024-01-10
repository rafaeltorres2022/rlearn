import numpy as np
from scipy.special import softmax
from rlearn.activation_functions import activation_factory

class MLPClassifier:

    def __init__(self, learning_rate = 0.001, mini_batch=200, activation_functions = 'relu', tol = 0.001) -> None:
        self.lr = learning_rate
        self.activation_func = activation_factory(activation_functions)
        self.weights =[]
        self.bs = []
        self.history = {'loss': []}
        self.classes = None
        self.classes_one_hot = None
        self.decay = 0
        self.adam = None
        self.mini_batch = mini_batch
        self.tol = tol
        #------

    def initialize_weights(self, n_features, hidden_layers):
        N_NEURONS_LAYERS = hidden_layers
        self.weights = [np.random.rand(n_features, N_NEURONS_LAYERS[0])]
        self.bs = [np.random.rand(N_NEURONS_LAYERS[0])]
        for i in range(1, len(N_NEURONS_LAYERS)):
            self.weights.append(np.random.randn(N_NEURONS_LAYERS[i-1], N_NEURONS_LAYERS[i])) 
            self.bs.append(np.random.randn(N_NEURONS_LAYERS[i]))
            
    def multilabel_log_loss(self, y_true, y_pred):
        return -np.average(y_true * np.log(y_pred + 1e-8))
    
    def relu_prime(self, x):
        return 1*(x>0)
    
    def propagate(self, X):
        #zs = weighted sums
        #outs = activation(zs)
        zs = []
        outs = []
        #foward
        #input layer
        z1 = X.dot(self.weights[0]) + self.bs[0]
        a1 = self.activation_func(z1)
        zs.append(z1)
        outs.append(a1)

        #hidden layers
        for layer in range(1, len(self.weights)-1):
            layer_z = outs[-1].dot(self.weights[layer]) + self.bs[layer]
            activation = self.activation_func(layer_z)

            zs.append(layer_z)
            outs.append(activation)

        #output layer
        zout = outs[-1].dot(self.weights[-1]) + self.bs[-1]
        aout = softmax(zout, axis=1)

        zs.append(zout)
        outs.append(aout)

        return zs, outs
    
    def predict(self, X):
        zs, outs = self.propagate(X)
        result = np.argmax(outs[-1], axis=1, keepdims=True)
        result = [self.classes[i][0] for i in result]
        return result
    
    def update_weights(self, gradients_w, gradients_b, epoch):
        #update weights
        gradients_w.reverse()
        gradients_b.reverse()

        # for g, w in zip(gradients_w, weights):
        #     print(g.shape, w.shape)
        for layer in range(len(self.weights)):
            # self.weights[layer] -= gradients_w[layer] * self.lr
            # self.bs[layer] -= gradients_b[layer] * self.lr
            self.weights[layer], self.bs[layer] = self.adam.step(self.weights[layer], self.bs[layer],
                                                            gradients_w[layer], gradients_b[layer], epoch, layer)

    def fit(self, X, y, hidden_layers, epochs):
        erros = []
        self.initialize_weights(X.shape[1], hidden_layers)
        self.adam = Adam2d(len(hidden_layers))
        self.classes = np.unique(y)
        self.classes_one_hot = np.eye(np.max(y)+1)[y]

        for epoch in range(epochs):
            X_batch, y_batch = self.define_batch(X, y)
            k_max = np.eye(np.max(y_batch)+1)[y_batch]

            zs, outs = self.propagate(X_batch)

            layer_erro = outs[-1] - k_max
            self.history['loss'].append(self.multilabel_log_loss(k_max, outs[-1]))
            if self.should_stop():
                break
            # backpropagation
            gradients_w = []
            gradients_b = []

            #output gradient
            derro_wout = outs[-2].T.dot(layer_erro)
            gradient_bout = layer_erro.sum(axis=0)

            gradients_w.append(derro_wout)
            gradients_b.append(gradient_bout)

            #print(a1.shape, layer_erro.shape, derro_wout.shape)


            #hidden layers gradient
            for layer in range(-2, -len(self.weights), -1): 
                layer_erro = layer_erro.dot(self.weights[layer+1].T) * self.relu_prime(zs[layer])
                derro_w = outs[layer-1].T.dot(layer_erro)
                gradient_b = layer_erro.sum(axis=0)

                gradients_w.append(derro_w)
                gradients_b.append(gradient_b)
            
            #input layer gradient
            layer_erro = layer_erro.dot(self.weights[1].T) * self.relu_prime(zs[0])
            derro_w = X_batch.T.dot(layer_erro)
            gradient_b = layer_erro.sum(axis=0)

            gradients_w.append(derro_w)
            gradients_b.append(gradient_b)
            self.update_weights(gradients_w, gradients_b, epoch)

    def define_batch(self, X, y):
        indexes = np.random.randint(0, high=X.shape[0], size=self.mini_batch)
        try:
            batch_x = X[indexes, :]
            batch_y = y[indexes,]
        except IndexError:
            batch_x = X[indexes]
            batch_y = y[indexes]
        return batch_x, batch_y
    
    def should_stop(self):
        try:
            should_stop = self.history[-1] > (self.history[-2] - self.tol)
        except KeyError:
            return False
        return should_stop

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

    def step(self, old_w, old_b, gradients_w, gradients_b, epoch, layer):
        epoch += 1
        self.mtw[layer] = (self.beta1 * self.mtw[layer]) + ((1 - self.beta1) * gradients_w)
        self.mtb[layer] = (self.beta1 * self.mtb[layer]) + ((1 - self.beta1) * gradients_b)
        self.vtw[layer] = self.beta2 * self.vtw[layer] + (1 - self.beta2) * gradients_w**2
        self.vtb[layer] = self.beta2 * self.vtb[layer] + (1 - self.beta2) * gradients_b**2
        m_hatw = self.mtw[layer] / (1 - self.beta1 ** epoch)
        m_hatb = self.mtb[layer] / (1 - self.beta1 ** epoch)
        v_hatw = self.vtw[layer] / (1 - self.beta2 ** epoch)
        v_hatb = self.vtb[layer] / (1 - self.beta2 ** epoch)
        new_w = old_w - self.lr * m_hatw / (np.sqrt(v_hatw) + self.ep)
        new_b = old_b - self.lr * m_hatb / (np.sqrt(v_hatb) + self.ep)
        return new_w, new_b
    
