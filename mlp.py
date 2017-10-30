import numpy as np
from util import *
from func import *
import queue as q


## Multi-Layer Perceptron
# (abstract base class)

class MLP():

    def __init__(self, dims, functions, distrib, model_ID=-1, split_ID=-1):
        assume(len(dims)-1 == len(functions), 'Invalid number of functions.')
        assume(all(f in {'sigmoid', 'sig', 'tanh', 'linear', 'lin', 'softmax'} for f in functions), 'Invalid function name.')
        assume(distrib[0] in {'uniform', 'normal'}, 'Invalid distribution form.')
        self.nlayers = len(dims)
        self.dims = dims
        self.weights = self.initialize_weights(self.random_distrib(distrib[0]), distrib[1])
        self.functions = list(self.assign_function(i) for i in functions)
        self.model_ID = model_ID
        self.split_ID = split_ID



    def initialize_weights(self, distrib_func, distrib_scale):
        return list(scale(normalize(distrib_func(self.dims[i + 1], self.dims[i] + 1)),
                          distrib_scale[0], distrib_scale[1]) for i in range(self.nlayers - 1))

    def random_distrib(self, d):
        dist_switch = {
            'uniform': np.random.rand,
            'normal': np.random.randn,
        }
        return dist_switch.get(d, np.random.rand)


    def assign_function(self, f):
        func_switch = {
            'sigmoid': [logsig, dlogsig],
            'sig': [logsig, dlogsig],
            'tanh': [tanh, dtanh],
            'linear': [linear, dlinear],
            'lin': [linear, dlinear],
            'softmax': [softmax, dsoftmax],
        }
        return func_switch.get(f, [logsig, logsig])


    ## forward pass
    # (single input vector)

    def forward(self, x):
        ins = []
        outs = []
        xx = augment(x)
        for i in range(self.nlayers-1):
            a = self.weights[i] @ xx
            h = augment(self.functions[i][0](a))
            xx = h
            if i == self.nlayers-2:
                h = h[:-1] # final layer outputs == predictions - remove bias (last one)
            ins.append(a)
            outs.append(h)
        return ins, outs




    ## forward & backprop pass
    # (single input and target vector)

    def backward(self, x, d):
        ins, outs = self.forward(x)
        y = outs[-1]
        dWs = []
        gg = None
        for i in reversed(range(self.nlayers-1)):
            if i == self.nlayers-2 : #the last layer == output
                gg = (d - y) * self.functions[i][1](ins[i])
                dW = outs[i-1].reshape((1,-1)).T * gg.reshape((1,-1))
                dWs.append(dW)
            elif i == 0 : #the first layer == input
                gg = (gg.T @ self.weights[i+1][:,0:-1]) * self.functions[i][1](ins[i])
                dW = augment(x).reshape((1,-1)).T * gg.reshape((1,-1))
                dWs.append(dW)
            else : #the layers between
                gg = (gg.T @ self.weights[i + 1][:, 0:-1]) * self.functions[i][1](ins[i])
                dW = outs[i-1].reshape((1,-1)).T * gg.reshape((1,-1))
                dWs.append(dW)

        return y, list(reversed(dWs))
