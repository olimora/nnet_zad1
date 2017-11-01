import atexit
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # todo: remove or change if not working
import matplotlib.pyplot as plt
import time

def assume(bool, msg):
    if not bool :
        raise Exception(msg)


#mat must be normalized to [0 .. 1]
def scale(mat, min, max):
    return mat * (max - min) + min

def normalize(mat): # classic 0 to 1
    return (mat - np.amin(mat)) / (np.amax(mat) - np.amin(mat))

def normalize2(mat): # std
    return (mat - np.mean(mat)) / np.std(mat)

def get_normalize_func(t):
    norm_switch = {
        'abs': normalize,
        'std': normalize2,
    }
    return norm_switch.get(t, normalize)

def labels_to_nums(labels):
    return ord(labels)-65
labels_to_nums = np.vectorize(labels_to_nums)


def plot_scatter(title, data_x, data_y, c):
    plt.scatter(data_x, data_y, c=c)
    plt.show(block=True)

## utility

def augment(X):
    if X.ndim == 1:
        return np.concatenate((X, [1]))
    else:
        pad = np.ones((1, X.shape[1]))
        return np.concatenate((X, pad), axis=0)


def onehot_decode(X):
    return np.argmax(X, axis=0)


def onehot_encode(L, c):
    if isinstance(L, int):
        L = [L]
    n = len(L)
    out = np.zeros((c, n))
    out[L, range(n)] = 1
    return np.squeeze(out)


## plotting

palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']


def limits(values, gap=0.05):
    x0 = np.min(values)
    x1 = np.max(values)
    xg = (x1 - x0) * gap
    return np.array((x0 - xg, x1 + xg))


def plot_errors(title, errors, test_error=None, block=True):
    plt.figure(1)
    plt.clf()

    plt.plot(errors)

    if test_error:
        plt.plot([test_error] * len(errors))

    plt.tight_layout()
    plt.gcf().canvas.set_window_title(title)
    plt.show(block=block)


def plot_both_errors(trainCEs, trainREs, testCE=None, testRE=None, pad=None, block=True):
    plt.figure(2)
    plt.clf()

    if pad is None:
        pad = max(len(trainCEs), len(trainREs))
    else:
        trainCEs = np.concatentate((trainCEs, [None] * (pad - len(trainCEs))))
        trainREs = np.concatentate((trainREs, [None] * (pad - len(trainREs))))

    plt.subplot(2, 1, 1)
    plt.title('Classification accuracy')
    plt.plot(100 * np.array(trainCEs))

    if testCE is not None:
        plt.plot([100 * testCE] * pad)

    plt.subplot(2, 1, 2)
    plt.title('Model loss (MSE)')
    plt.plot(trainREs)

    if testRE is not None:
        plt.plot([testRE] * pad)

    plt.tight_layout()
    plt.gcf().canvas.set_window_title('Errors')
    plt.show(block=block)


def plot_dots(inputs, targets=None, predicted=None, s=60, i_x=0, i_y=1, block=True):
    plt.figure(3)
    plt.clf()

    if targets is None:
        plt.gcf().canvas.set_window_title('Data distribution')
        plt.scatter(inputs[i_x, :], inputs[i_y, :], s=s, c=palette[-1], edgecolors=[0.4] * 3, alpha=0.5)

    elif predicted is None:
        plt.gcf().canvas.set_window_title('Class distribution')
        for i, c in enumerate(set(targets)):
            plt.scatter(inputs[i_x, targets == c], inputs[i_y, targets == c], s=s, c=palette[i], edgecolors=[0.4] * 3)

    else:
        plt.gcf().canvas.set_window_title('Predicted vs. targets')
        for i, c in enumerate(set(targets)):
            plt.scatter(inputs[i_x, targets == c], inputs[i_y, targets == c], s=2.0 * s, c=palette[i], edgecolors=None,
                        alpha=0.333)

        for i, c in enumerate(set(targets)):
            plt.scatter(inputs[i_x, predicted == c], inputs[i_y, predicted == c], s=0.5 * s, c=palette[i],
                        edgecolors=None)

    plt.xlim(limits(inputs[i_x, :]))
    plt.ylim(limits(inputs[i_y, :]))
    plt.tight_layout()
    plt.show(block=block)


def plot_reg_density(title, inputs, targets, outputs=None, s=70, block=True):
    plt.figure(4, figsize=(9, 9))
    plt.clf()

    if outputs is not None:
        plt.subplot(2, 1, 2)
        plt.title('Predicted')
        plt.scatter(inputs[0], inputs[1], s=s * outputs)

        plt.subplot(2, 1, 1)
        plt.title('Original')

    plt.scatter(inputs[0], inputs[1], s=s * targets)
    plt.gcf().canvas.set_window_title(title)
    plt.tight_layout()

    plt.show(block=block)


## interactive drawing, very fragile....

wait = 0.0


def clear():
    plt.clf()


def ion():
    plt.ion()
    plt.show()
    time.sleep(wait)


def ioff():
    plt.ioff()
    plt.close()


def redraw():
    plt.gcf().canvas.draw()
    time.sleep(wait)


## non-blocking figures still block at end

def finish():
    plt.show(block=True)  # block until all figures are closed


atexit.register(finish)
