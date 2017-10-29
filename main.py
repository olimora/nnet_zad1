import numpy as np
import random
from util import *
from classifier import *
import multiprocessing as mp
from itertools import repeat

def parallel_cross_validation(split_id, passed_data):
    #passed data from common main
    split = passed_data[0]
    train_inputs = passed_data[1]
    train_labels = passed_data[2]

    #indexes of validation entries are in dedicated split
    valid_ind = split[split_id]
    #indexes of estimation entries are all the others splits combined
    #so concantenate indexes from list 'split' on indexes 0:10 without split_id
    estim_ind = np.concatenate(split[np.delete(np.arange(10), split_id, axis=0)])
    estim_inputs = train_inputs[:, estim_ind]
    estim_labels = train_labels[estim_ind]
    valid_inputs = train_inputs[:, valid_ind]
    valid_labels = train_labels[valid_ind]

    ## train & validate
    model = MLPClassifier([train_inputs.shape[0], 20, 6, np.max(train_labels) + 1],
                          ['tanh', 'sig', 'lin'], ['uniform', [0, 1]],
                          model_ID=split_id)
    trainCEs, trainREs = model.train(estim_inputs, estim_labels, valid_inputs, valid_labels,
                                     alpha=0.05, momentum=0.1,
                                     min_accuracy=97, max_epoch=500, min_delay_expectancy=50,
                                     q_size=30, raised_err_threashold=0.66, acc_err_threshold=0.001,
                                     trace=False, trace_interval=10)
    validCE, validRE = model.test(valid_inputs, valid_labels)

    return np.array([split_id, validCE, validRE])

if __name__ == '__main__':

    ## load data
    train_inputs_org = np.loadtxt('2d.trn.dat', skiprows=1, usecols=(0, 1)).T
    train_labels_org = np.loadtxt('2d.trn.dat', dtype='S20', skiprows=1, usecols=(2)).astype(str).T
    train_labels_org = labels_to_nums(train_labels_org)
    test_inputs_org = np.loadtxt('2d.tst.dat', skiprows=1, usecols=(0, 1)).T
    test_labels_org = np.loadtxt('2d.tst.dat', dtype='S20', skiprows=1, usecols=(2)).astype(str).T
    test_labels_org = labels_to_nums(test_labels_org)

    # plot_scatter("training data", test_inputs[0], test_inputs[1], test_labels)

    # matrix of experiment setups
    hyperparameters = list()
    hyperparameters.append(
        ['abs', # data normalization type
        [2, 20, 6, 3], ['tanh', 'sig', 'lin'], ['uniform', [0, 1]], # layers, functions, distribution and scale
        0.05, 0.1,          # aplha, momentum
        97, 500, 50,        # min_accuracy, max_epoch, min_delay_expectancy
        30, 0.66, 0.001])   # q_size, raised_err_threashold, acc_err_threshold
    hyperparameters.append(
        ['std', # data normalization type
        [2, 12, 6, 3], ['sig', 'sig', 'sig'], ['normal', [-1, 1]], # layers, functions, distribution and scale
        0.05, 0.05,         # aplha, momentum
        97, 500, 50,        # min_accuracy, max_epoch, min_delay_expectancy
        20, 0.66, 0.001])   # q_size, raised_err_threashold, acc_err_threshold


    # loop the models setups
    for i in range(len(hyperparameters)):
        norm_func = get_normalize_func(hyperparameters[i][0])
        train_inputs = train_inputs_org
        train_inputs[0] = norm_func(train_inputs[0])
        train_inputs[1] = norm_func(train_inputs[1])
        train_labels = train_labels_org
        # test_inputs = test_inputs_org
        # test_inputs[0] = norm_func(test_inputs[0])
        # test_inputs[1] = norm_func(test_inputs[1])

        # split to 10 sets
        ind = np.arange(len(train_labels_org))
        random.shuffle(ind)
        split = np.array(np.split(ind, 10))

        # parallel cross validation
        pool = mp.Pool(processes=4)  # removing processes argument makes the code run on all available cores
        results = np.array(
            pool.starmap(parallel_cross_validation, zip(np.arange(10), repeat([split, train_inputs, train_labels]))))
        print(results)

        # mean from results

        # save hyperparameters and results in csv

    print(1 / 0)


    # read csv, get best hyperparameters, create model like that and test on test data

    # normalize data

    # create model and train
    model = MLPClassifier([train_inputs.shape[0], 20, 6, np.max(train_labels) + 1],
                          ['tanh', 'sig', 'lin'], ['uniform', [0, 1]],
                          model_ID=0)
    trainCEs, trainREs = model.train(estim_inputs, estim_labels, valid_inputs, valid_labels,
                                     alpha=0.05, momentum=0.1,
                                     min_accuracy=97, max_epoch=500, min_delay_expectancy=50,
                                     q_size=30, raised_err_threashold=0.66, acc_err_threshold=0.001,
                                     trace=False, trace_interval=10)

    # test and write to csv
    testCE, testRE = model.test(test_inputs, test_labels)
    print('Final testing error: CE = {:6.2%}, RE = {:.5f}'.format(testCE, testRE))

    plot_both_errors(trainCEs, trainREs, testCE, testRE, block=False)

    # calculate a confusion matrix.




