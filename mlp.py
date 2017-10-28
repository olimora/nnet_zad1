import numpy as np
from util import *
from func import *


## Multi-Layer Perceptron
# (abstract base class)

class MLP():

    def __init__(self, dims, functions):
        assert len(dims)-1 == len(functions)
        self.nlayers = len(dims)
        self.dims = dims
        self.func_switch = {
            'sigmoid': [logsig, dlogsig],
            'sig': [logsig, dlogsig],
            'tanh': [tanh, dtanh],
            'linear': [linear, dlinear],
            'lin': [linear, dlinear],
            'softmax': [softmax, dsoftmax],
        }
        self.functions = list(self.assign_function(i) for i in functions)
        self.weights = list(np.random.rand(dims[i+1], dims[i]+1) for i in range(self.nlayers - 1))



    def assign_function(self, f):
        return self.func_switch.get(f, [logsig, logsig])


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
                dW = outs[i-1].reshape((1,-1)).T @ gg.reshape((1,-1))
                dWs.append(dW)
            elif i == 0 : #the first layer == input
                gg = (gg.T @ self.weights[i+1][:,0:-1]) * self.functions[i][1](ins[i])
                dW = augment(x).reshape((1,-1)).T @ gg.reshape((1,-1))
                dWs.append(dW)
            else : #the layers between
                gg = (gg.T @ self.weights[i + 1][:, 0:-1]) * self.functions[i][1](ins[i])
                dW = outs[i-1].reshape((1,-1)).T @ gg.reshape((1,-1))
                dWs.append(dW)

        return y, list(reversed(dWs))
