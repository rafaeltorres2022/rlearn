import numpy as np
from rlearn.activation_functions import *
from rlearn.solvers import Adam2d
from numpy.lib.stride_tricks import sliding_window_view
from time import time

class Layer:

    def __init__(self) -> None:
        self.input_dim = None
        self.layer_error = None
        self.layer_input = None

    def set_input_dim(self, input_dim : tuple):
        self.input_dim = input_dim

    def set_layer_error(self, error):
        self.layer_error = error

class Conv3C(Layer):
    def __init__(self, kernel_size = 3, n_chanels_kernel = 3, out_channels = 3, stride = 1, activation=LeakyRelu()) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels
        self.weights = np.array(
            [np.random.normal(0, 1, (kernel_size, kernel_size, n_chanels_kernel)) for _ in range(out_channels)]
            )
        self.biases = np.zeros(out_channels)
        self.activation = activation
        #self.cache = None


    # def convolve(self, image, kernel, w_shape, h_shape):
    #     conv = np.array([])
    #     for row in range(self.kernel_size, image.shape[0]+1, self.stride):
    #         for col in range(self.kernel_size, image.shape[1]+1, self.stride):
    #             stride_content = image[row-self.kernel_size:row, col-self.kernel_size : col]
    #             conv = np.app#end(conv, np.vdot(stride_content, np.rot90(kernel, 2)))
    #     return conv.reshape(w_shape, h_shape)

    def foward(self, imagem):
        #start = time()
        '''
        k = number of filters i.e: number of channels on the resulting image
        i and j = filter dimensions
        c = filter channels
        n = number of inputs
        w and h = dimensions of the output
        c = input channels
        i and j = strides to be filtered
        '''
        self.layer_input = imagem
        strides = sliding_window_view(imagem, window_shape=(self.kernel_size,self.kernel_size,), axis=(1,2,))[:, 0::self.stride, 0::self.stride]
        conv = np.einsum('kijc,nwhcij->nwhk',self.weights, strides, optimize='optimize')
        for c in range(len(self.biases)):
            conv[:,:,:,c] += self.biases[c]
        #end = time()
        #print(f'Foward Conv3C: {#end-#start}')
        return self.activation.activate(conv)
    
    
    def get_output_dim(self):
        return  (int((self.input_dim[1] - self.kernel_size) / self.stride+1),
            int((self.input_dim[2] - self.kernel_size) / self.stride+1),
            self.out_channels)
    
    def backward(self, prev_layer_error, weights_prev_layer, solver, layer):
        # print(self.__class__, prev_layer_error.shape)
        # print(self.__class__, self.weights.shape)
        #patches = sliding_window_view(self.relu_prime(self.layer_input), window_shape=(prev_layer_error.shape[1],prev_layer_error.shape[2],), axis=(1,2,))[:, 0::self.stride, 0::self.stride]
        #start = time()
        self.layer_input = self.activation.prime(self.layer_input)
        back_conv_w = np.zeros(shape=self.weights.shape)
        k_s = prev_layer_error.shape[1]
        for row in range(back_conv_w.shape[1]):
            for col in range(back_conv_w.shape[2]):
                for channel in range(back_conv_w.shape[3]):
                    new_value = np.einsum('nij, nijk -> ', self.layer_input[:,row:row+k_s, col:col+k_s, channel], prev_layer_error)
                    back_conv_w[:, row, col, channel] += new_value
        # self.cache = (prev_layer_error, patches)
        # back_conv_w = np.einsum('kijc,nwhdij->cwhd',prev_layer_error, patches, optimize='optimize')
        #end = time()
        #print(f'Backward Conv3C wrt Input: {#end-#start}')
        back_bs = prev_layer_error.sum(axis=(-4, -3, -2))
        # print(back_conv_w.shape)
        # print(back_bs.shape)
        # print(back_conv_w.shape)
        self.weights, self.biases = solver.step(self.weights, self.biases, back_conv_w, back_bs, layer)
        #start = time()
        test = np.zeros(shape=self.layer_input.shape)
        k_s = self.weights.shape[1]
        for row in range(prev_layer_error.shape[1]):
            for col in range(prev_layer_error.shape[2]):
                test[:,row:row+k_s, col:col+k_s, :] += np.einsum('kijc, xk -> xijc', self.weights, prev_layer_error[:, row, col, :])
        #end = time()
        #print(f'Backward Conv3C wrt Weights: {#end-#start}')
        return test
    
class FC(Layer):
    def __init__(self, n_neurons, activation=LeakyRelu(), is_output_layer = False) -> None:
        self.activation = activation
        self.n_neurons = n_neurons
        self.weights = None
        self.bias = None
        self.last_z = None
        self.last_out = None
        self.last_input = None
        self.is_output_layer = is_output_layer

    def set_input_dim(self, input_dim : tuple):
        self.input_dim = input_dim
        self.initialize_weights(self.input_dim[-1])
        
    def initialize_weights(self, n_features):
        self.weights = np.random.rand(n_features, self.n_neurons)
        self.bias = np.zeros(self.n_neurons)

    def foward(self, input):
        self.last_input = input
        z = input.dot(self.weights) + self.bias
        self.last_z = z
        out = self.activation.activate(z)
        self.last_out = out
        return out
    
    def backward(self, prev_layer_error, weights_prev_layer, solver, layer):
        if self.is_output_layer == True:
            layer_erro = prev_layer_error
            gradient_w = self.last_input.T.dot(prev_layer_error)
        else:
            layer_erro = prev_layer_error.dot(weights_prev_layer.T) * self.activation.prime(self.last_z)
            gradient_w = self.last_input.T.dot(layer_erro)
        gradient_b = layer_erro.sum(axis=0)
        self.weights, self.bias = solver.step(self.weights, self.bias, gradient_w, gradient_b, layer)
        return layer_erro
    
    def get_output_dim(self):
        return (self.weights.shape[1],)
    
class Squeezing(Layer):

    def foward(self, input):
        return input.reshape(len(input), -1)
    
    def get_output_dim(self):
        return (self.input_dim[1] * self.input_dim[2] * self.input_dim[3], )
    
    def backward(self, prev_layer_error, weights_prev_layer, solver, layer):
        return prev_layer_error.dot(weights_prev_layer.T).reshape(-1, *self.input_dim[1:])
    
class MaxPooling(Layer):
    def __init__(self, size=2, stride=2) -> None:
        self.size = size
        self.stride = stride
        #self.test_stuff = []

    def foward(self, imagem, ):
        #start = time()
        self.layer_input = imagem
        pool = sliding_window_view(imagem, window_shape=(self.size,self.size,), axis=(1,2,))[:, 0::self.stride, 0::self.stride]
        #end = time()
        #print(f'Foward Pooling: {#end-#start}')
        return np.max(pool, axis=(-2,-1))
    
    def get_output_dim(self):
        return ( 
            int((self.input_dim[1] - self.size) / self.stride+1),
            int((self.input_dim[2] - self.size) / self.stride+1),
            self.input_dim[3])
    
    
    def backward(self, prev_layer_error, weights_prev_layer, solver, layer):
        #start = time()
        patches = sliding_window_view(self.layer_input, window_shape=(self.size,self.size,), axis=(1,2,))[:, 0::self.stride, 0::self.stride]
        #self.test_stuff.app#end(prev_layer_error)
        # self.test_stuff.app#end(patches)
        temp_patches = np.copy(patches)
        for item in range(temp_patches.shape[0]):
                for row in range(temp_patches.shape[1]):
                        for col in range(temp_patches.shape[2]):
                                for channel in range(temp_patches.shape[3]):
                                        temp = temp_patches[item][row][col][channel]
                                        value = np.amax(temp, axis=(-2,-1))
                                        #index = np.unravel_index(index, temp_patches[:][0][0][row][0].shape)
                                        # print(temp)
                                        # print(temp.shape)
                                        # print(value)
                                        index = np.argwhere(temp == value)[0]
                                        # print(index)
                                        # print(temp[tuple(index)])
                                        temp[:] = 0
                                        temp[tuple(index)] = 1
                                        # print(temp)
        #self.test_stuff.app#end(temp_patches)
        test_einsum = np.einsum('nijc, nijcxy -> nijcxy', prev_layer_error, temp_patches)
        test_einsum = test_einsum.reshape(-1, *self.input_dim[1:])
        #end = time()
        #print(f'Backward Pooling: {#end-#start}')
        return test_einsum
    
class NNModel:

    def __init__(self, input_dim : tuple, layers = [], learning_rate=0.001) -> None:
        self.layers = layers
        self.in_out_dimensions = self.define_output_dim(input_dim)
        self.output_classes = None
        self.solver = Adam2d(len(self.layers), learning_rate=learning_rate)
        self.layers[-1].is_output_layer = True
        self.history = {'train_loss' : [], 'train_accuracy' : [], 'test_loss' : [], 'test_accuracy' : []}

    def define_output_dim(self, input_dim):
        dims = []
        prev_output = (None, *input_dim)
        for layer in self.layers:
            inputs = prev_output
            layer.set_input_dim(prev_output)
            prev_output = (None, *layer.get_output_dim())
            dims.append((layer.__class__, inputs, prev_output))
        return dims

    def foward(self, input):
        layer_output = input
        for layer in self.layers:
            layer_output = layer.foward(layer_output)

        return layer_output
    
    def backward(self, error):
        prev_layer_weights = 1
        prev_layer_error = error
        for i, layer in enumerate(self.layers[::-1]):
            prev_layer_error = layer.backward(prev_layer_error, prev_layer_weights, self.solver, -i-1)
            if layer.__class__ == Squeezing: continue
            if layer.__class__ == MaxPooling: continue
            prev_layer_weights = layer.weights

    def fit(self, X, y, X_test = None, y_test = None, epochs=1000, batch_size=200, verbose=1, error_threshold = 0.001, acc_threshold = 0.9):
        self.output_classes = np.unique(y)
        for epoch in range(1, epochs+1):
            #batch_X, batch_y = self.define_batch(X, y, batch_size=np.minimum(len(X), batch_size))
            for batch_index in range(0, len(X), batch_size):
                #print(f'Batch {batch_index}')
                batch_X = X[batch_index:batch_index+batch_size]
                batch_y = y[batch_index:batch_index+batch_size]
                y_one_hot_encoded = self.encode_y(batch_y)
                result = self.foward(batch_X)
                error = result - y_one_hot_encoded
                self.backward(error)

            y_one_hot_encoded_all_set = self.encode_y(y)
            result_all_set = self.foward(X)


            self.history['train_loss'].append(self.multilabel_log_loss(y_one_hot_encoded_all_set, result_all_set))
            self.history['train_accuracy'].append(self.model_accuracy(y, self.predict(X)))

            if (X_test is not None):
                y_one_hot_encoded_all_set_test = self.encode_y(y_test)
                result_all_set_test = self.foward(X_test)
                self.history['test_loss'].append(self.multilabel_log_loss(y_one_hot_encoded_all_set_test, result_all_set_test))
                self.history['test_accuracy'].append(self.model_accuracy(y_test, self.predict(X_test)))
            if (epoch % verbose == 0) or (epoch == 1):
                print(f'Epoch {epoch}: Training Loss {"%.2f" % self.history["train_loss"][-1]}\tTraining Accuracy {"%.2f" % self.history["train_accuracy"][-1]}\tTest Loss {"%.2f" % self.history["test_loss"][-1]}\tTest Accuracy {"%.2f" % self.history["test_accuracy"][-1]}')
            if (self.history['test_loss'][-1] < error_threshold): 
                print(f'Epoch {epoch}: Training Loss {"%.2f" % self.history["train_loss"][-1]}\tTraining Accuracy {"%.2f" % self.history["train_accuracy"][-1]}\tTest Loss {"%.2f" % self.history["test_loss"][-1]}\tTest Accuracy {"%.2f" % self.history["test_accuracy"][-1]}')
                break
            if (self.history['test_accuracy'][-1] > acc_threshold): 
                print(f'Epoch {epoch}: Training Loss {"%.2f" % self.history["train_loss"][-1]}\tTraining Accuracy {"%.2f" % self.history["train_accuracy"][-1]}\tTest Loss {"%.2f" % self.history["test_loss"][-1]}\tTest Accuracy {"%.2f" % self.history["test_accuracy"][-1]}')
                break

        for l in self.layers:
            l.layer_input = None


    def predict(self, X, get_probabilities=False):
        result = self.foward(X)
        if get_probabilities: return result
        result = np.argmax(result, axis=1, keepdims=True)
        result = [self.output_classes[i][0] for i in result]
        return result
    
    def encode_y(self, y):
        return np.eye(np.max(y)+1)[y]
    
    def model_accuracy(self, y_true, pred):
        return (y_true == pred).sum() / len(y_true)

    def multilabel_log_loss(self, y_true, y_pred):
        return -np.average(y_true * np.log(y_pred + 1e-8))
    
    def define_batch(self, X, y, batch_size):
        indexes = np.random.randint(0, high=X.shape[0], size=batch_size)
        try:
            batch_x = X[indexes, :]
            batch_y = y[indexes,]
        except IndexError:
            batch_x = X[indexes]
            batch_y = y[indexes]
        return batch_x, batch_y
    
