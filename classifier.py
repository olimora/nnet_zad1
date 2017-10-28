import numpy as np

from mlp import *
from util import *


class MLPClassifier(MLP):

    def __init__(self, dims, functions):
        self.n_classes = dims[-1]
        super().__init__(dims, functions)


    def cost(self, targets, outputs): # new
        return np.sum((targets - outputs)**2, axis=0)

    ## prediction pass

    def predict(self, inputs):
        # outputs, *_ = self.forward(inputs)  # if self.forward() can take a whole batch
        _, outputs = self.forward(inputs)
        outputs = outputs[-1]
        # outputs = np.stack([self.forward(x)[0] for x in inputs.T]) # otherwise
        return onehot_decode(outputs)


    ## testing pass

    def test(self, inputs, labels):
        # outputs, *_ = self.forward(inputs) # FIXME
        _, outputs = self.forward(inputs)
        outputs = outputs[-1]
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
            CE = 0
            RE = 0

            for i in np.random.permutation(count):
                x = inputs[:, i] # FIXME
                d = targets[:, i] # FIXME

                y, dWs = self.backward(x, d)

                CE += labels[i] != onehot_decode(y)
                RE += self.cost(d,y)

                for i in range(self.nlayers-1):
                    self.weights[i] += alpha*dWs[i].T

            CE /= count
            RE /= count

            CEs.append(CE)
            REs.append(RE)

            # if (ep+1) % 10 == 0:
            #     print('Model {:1d}: Ep {:3d}/{}: '.format(model_num, ep+1, eps), end='')
            #     print('CE = {:6.2%}, RE = {:.5f}'.format(CE, RE))

            print('Model {:1d}: Ep {:3d}/{}: '.format(model_num, ep + 1, eps), end='')
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
