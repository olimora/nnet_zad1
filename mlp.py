import numpy as np

from util import *


## Multi-Layer Perceptron
# (abstract base class)

class MLP():

    def __init__(self, dim_in, dim_hid, dim_out):
        self.dim_in     = dim_in
        self.dim_hid    = dim_hid
        self.dim_out    = dim_out

        self.W_hid = np.random.rand(dim_hid, dim_in+1) # FIXME
        self.W_out = np.random.rand(dim_out, dim_hid+1) # FIXME


    ## activation functions & derivations
    # (not implemented, to be overriden in derived classes)

    def f_hid(self, x):
        pass 

    def df_hid(self, x):
        pass 

    def f_out(self, x):
        pass 

    def df_out(self, x):
        pass 


    ## forward pass
    # (single input vector)

    def forward(self, x):
        a = self.W_hid @ augment(x) # FIXME
        h = augment(self.f_hid(a)) # FIXME
        b = self.W_out @ h # FIXME
        y = self.f_out(b) # FIXME

        return y, b, h, a


    ## forward & backprop pass
    # (single input and target vector)

    def backward(self, x, d):
        y, b, h, a = self.forward(x)

        g_out = (d - y) * self.df_out(b) # FIXME
        g_hid = (g_out.T @ self.W_out[:,0:-1]) * self.df_hid(a) # FIXME

        dW_out = h.reshape((1,-1)).T @ g_out.reshape((1,-1)) # FIXME
        dW_hid = augment(x).reshape((1,-1)).T @ g_hid.reshape((1,-1)) # FIXME

        return y, dW_hid, dW_out
