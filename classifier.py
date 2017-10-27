import numpy as np

from mlp import *
from util import *


class MLPClassifier(MLP):

    def __init__(self, dim_in, dim_hid, n_classes):
        self.n_classes = n_classes
        super().__init__(dim_in, dim_hid, dim_out=n_classes)

    
    ## functions
    # https://en.wikipedia.org/wiki/Activation_function
    # sigmoid == logistic == soft step

    def cost(self, targets, outputs): # new
        return np.sum((targets - outputs)**2, axis=0)

    def f_hid(self, x): # override
        # FIXME sigmoid or tanh
        return 1 / (1 + (np.e ** (-x)))  # sigmoid
        # return (2 / (1 + (np.e ** (-2*x)))) - 1  # TanH

    def df_hid(self, x): # override
        # FIXME corresponding derivation
        return self.f_hid(x) * (1 - self.f_hid(x))  # sigmoid
        # return 1 - (self.f_hid(x)**2) # TanH

    def f_out(self, x): # override
        # FIXME sigmoid or softmax # try binary step
        return 1 / (1 + (np.e ** (-x)))  # sigmoid 8.00%, RE = 0.28998;6.00%, RE = 0.26435
        # return np.exp(x) / np.sum(np.exp(x)) # softmax # ,axis = 0 #subtract max? https://stackoverflow.com/questions/34968722/softmax-function-python

    def df_out(self, x): # override
        # FIXME corresponding derivation
        return self.f_hid(x) * (1 - self.f_hid(x))  # sigmoid
        # softmax https://stackoverflow.com/questions/36279904/softmax-derivative-in-numpy-approaches-0-implementation


    ## prediction pass

    def predict(self, inputs):
        outputs, *_ = self.forward(inputs)  # if self.forward() can take a whole batch
        # outputs = np.stack([self.forward(x)[0] for x in inputs.T]) # otherwise
        return onehot_decode(outputs)


    ## testing pass

    def test(self, inputs, labels):
        outputs, *_ = self.forward(inputs) # FIXME
        targets = onehot_encode(labels, self.n_classes) # FIXME
        predicted = onehot_decode(outputs) # FIXME
        CE = np.sum(labels != predicted) / inputs.shape[1] # FIXME
        RE = np.sum(self.cost(targets,outputs)) / inputs.shape[1]# FIXME
        return CE, RE


    ## training

    def train(self, inputs, labels, alpha=0.1, eps=100, trace=False, trace_interval=10, model_num=-1):
        (_, count) = inputs.shape
        targets = onehot_encode(labels, self.n_classes)

        if trace:
            ion()

        CEs = []
        REs = []

        for ep in range(eps):
            print('Model {:1d}: Ep {:3d}/{}: '.format(model_num, ep+1, eps), end='')
            CE = 0
            RE = 0

            for i in np.random.permutation(count):
                x = inputs[:, i] # FIXME
                d = targets[:, i] # FIXME

                y, dW_hid, dW_out = self.backward(x, d)

                CE += labels[i] != onehot_decode(y)
                RE += self.cost(d,y)

                self.W_hid += alpha*dW_hid.T # FIXME
                self.W_out += alpha*dW_out.T # FIXME

            CE /= count
            RE /= count

            CEs.append(CE)
            REs.append(RE)

            print('CE = {:6.2%}, RE = {:.5f}'.format(CE, RE))

            if trace and ((ep+1) % trace_interval == 0):
                clear()
                predicted = self.predict(inputs)
                plot_dots(inputs, labels, predicted, block=False)
                plot_both_errors(CEs, REs, block=False)
                redraw()

        if trace:
            ioff()

        print()

        return CEs, REs
