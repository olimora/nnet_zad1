import numpy as np
import math as mth
from mlp import *
from util import *


class MLPClassifier(MLP):

    def __init__(self, dims, functions, distrib, model_ID = -1):
        self.n_classes = dims[-1]
        super().__init__(dims, functions, distrib, model_ID)


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

    def train(self, inputs, labels, validation_inputs=None, validation_labels=None,
              alpha=0.1, momentum=0,
              min_accuracy=95, max_epoch=500, min_delay_expectancy=50, q_size=10, raised_err_threashold=10, acc_err_threshold=1,
              trace=False, trace_interval=10):

        (_, count) = inputs.shape
        targets = onehot_encode(labels, self.n_classes)

        assume(momentum >= 0 and momentum <=1, 'Invalid momentum.')
        last_dWs = list((np.zeros((self.dims[z + 1], self.dims[z] + 1)).T for z in range(self.nlayers - 1)))

        self.best_weights = self.weights # weights from epoche with minimal validation error
        self.best_error = 100
        self.best_epoch = 0
        self.min_accuracy = min_accuracy
        self.max_epoch = max_epoch
        self.min_delay_expectancy = min_delay_expectancy
        self.errors_queue = q.Queue(q_size)
        self.raised_err_threashold = raised_err_threashold
        self.acc_err_threshold = acc_err_threshold

        if trace:
            ion()

        CEs = []
        REs = []

        for ep in range(self.max_epoch):
            CE = 0
            RE = 0

            for i in np.random.permutation(count):
                x = inputs[:, i]
                d = targets[:, i]

                y, dWs = self.backward(x, d)

                CE += labels[i] != onehot_decode(y)
                RE += self.cost(d,y)

                for j in range(self.nlayers-1):
                    self.weights[j] += alpha*dWs[j].T + momentum*last_dWs[j].T
                last_dWs = dWs

            CE /= count
            RE /= count

            CEs.append(CE)
            REs.append(RE)

            # if (ep+1) % 10 == 0:
            #     print('Model {:1d}: Ep {:3d}/{}: '.format(model_num, ep+1, eps), end='')
            #     print('CE = {:6.2%}, RE = {:.5f}'.format(CE, RE))

            print('Model {:1d}: Ep {:3d}/{}: '.format(self.model_ID, ep, self.max_epoch), end='')
            print('CE = {:6.2%}, RE = {:.5f}'.format(CE, RE), end='')

            if trace and ((ep+1) % trace_interval == 0):
                clear()
                predicted = self.predict(inputs)
                plot_dots(inputs, labels, predicted, block=False)
                plot_both_errors(CEs, REs, block=False)
                redraw()

            # early stopping
            term_min_delay, term_acc_err, term_raised_err, vCE, vRE = self.early_stopping(ep, validation_inputs, validation_labels)
            print(';')

            # consider terminating only if accuracy is bigger than minimal accuracy.
            # need to convert min accuracy to max classification error
            if vCE <= (100-self.min_accuracy)/100:
                if term_min_delay:
                    print('Training terminated in: Model = {:1d}, Epoch = {:d}, Best Epoch = {:d} '
                          'due inability to reach new minimum'.format(self.model_ID, ep, self.best_epoch))
                    self.weights = self.best_weights
                    break
                if term_acc_err:
                    print('Training terminated in: Model = {:1d}, Epoch = {:d}, Best Epoch = {:d} '
                          'due to accumulated error'.format(self.model_ID, ep, self.best_epoch))
                    self.weights = self.best_weights
                    break
                if term_raised_err:
                    print('Training terminated in: Model = {:1d}, Epoch = {:d}, Best Epoch = {:d} '
                          'due to raised error'.format(self.model_ID, ep, self.best_epoch))
                    self.weights = self.best_weights
                    break


        if trace:
            ioff()

        print()

        return CEs, REs


    def early_stopping(self, ep, validation_inputs, validation_labels):
        # validate net
        vCE, vRE = self.test(validation_inputs, validation_labels)

        # remembering the best weights
        if vRE < self.best_error:
            self.best_epoch = ep
            self.best_error = vRE
            self.best_weights = self.weights

        # keeping Q actual
        if self.errors_queue.full():
            self.errors_queue.get_nowait()
        self.errors_queue.put_nowait(vRE)

        # check the errors
        accumulated_error = 0
        raised_error = 0
        for qi in range(1, self.errors_queue.qsize()):
            delta_error = self.errors_queue.queue[qi] - self.errors_queue.queue[qi - 1]
            accumulated_error += delta_error
            if delta_error > 0:
                raised_error += 1
        if self.errors_queue.qsize() is not 0:
            accumulated_error /= self.errors_queue.qsize()
            raised_error /= self.errors_queue.qsize()

        print('; Validation: CE = {:6.2%}, RE = {:.5f}, Raised Err = {:1.2f}, Accumul Err = {:2.5f}'
              .format(vCE, vRE, raised_error, accumulated_error), end='')

        # terminate training if conditions is reached
        term_min_delay = False
        term_acc_err = False
        term_raised_err = False
        if ep - self.best_epoch > self.min_delay_expectancy:
            term_min_delay = True
        if accumulated_error > self.acc_err_threshold:
            term_acc_err = True
        if raised_error >= self.raised_err_threashold:
            term_raised_err = True

        return term_min_delay, term_acc_err, term_raised_err, vCE, vRE