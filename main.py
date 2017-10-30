import numpy as np
import random
from util import *
from classifier import *
import multiprocessing as mp
from itertools import repeat
import datetime
import csv

def parallel_cross_validation(split_ID, passed_data):
    #passed data from common main
    split = passed_data[0]
    train_inputs = passed_data[1]
    train_labels = passed_data[2]
    model_ID = passed_data[3]
    parameters = passed_data[4]

    #indexes of validation entries are in dedicated split
    valid_ind = split[split_ID]
    #indexes of estimation entries are all the others splits combined
    #so concantenate indexes from list 'split' on indexes 0:10 without split_id
    estim_ind = np.concatenate(split[np.delete(np.arange(10), split_ID, axis=0)])
    estim_inputs = train_inputs[:, estim_ind]
    estim_labels = train_labels[estim_ind]
    valid_inputs = train_inputs[:, valid_ind]
    valid_labels = train_labels[valid_ind]

    ## train & validate
    model = MLPClassifier(parameters[1], parameters[2], parameters[3],
                          model_ID=model_ID, split_ID=split_ID)
    trainCEs, trainREs, validCE, validRE, epochs = model.train(estim_inputs, estim_labels, valid_inputs, valid_labels,
                                     alpha=parameters[4],
                                     momentum=parameters[5],
                                     min_accuracy=parameters[6],
                                     max_epoch=parameters[7],
                                     min_delay_expectancy=parameters[8],
                                     q_size=parameters[9],
                                     raised_err_threashold=parameters[10],
                                     acc_err_threshold=parameters[11],
                                     trace=False, trace_interval=10)
    testCE, testRE = model.test(valid_inputs, valid_labels)
    return np.array([split_ID, testCE, testRE, epochs])

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
    # hyperparameters.append(
    #     ['std', # data normalization type
    #     [2, 12, 6, 3], ['sig', 'sig', 'sig'], ['normal', [-1, 1]], # layers, functions, distribution and scale
    #     0.05, 0.05,         # aplha, momentum
    #     97, 500, 50,        # min_accuracy, max_epoch, min_delay_expectancy
    #     20, 0.66, 0.001])   # q_size, raised_err_threashold, acc_err_threshold


    date_time = datetime.datetime.now()
    file_name = 'D://skola//NNET//source//zadanie1_results//validation_results_{:4d}_{:2d}_{:2d}__{:2d}_{:2d}_{:2d}.csv' \
        .format(date_time.year, date_time.month, date_time.day, date_time.hour, date_time.minute, date_time.second)
    file = open(file_name, 'w', newline='')
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['model', 'data_normalization', 'layers', 'functions', 'distribution',
                     'aplha', 'momentum', 'min_accuracy', 'max_epoch', 'min_delay_expectancy',
                     'q_size', 'raised_err_threashold', 'acc_err_threshold',
                     'mean_valid_CE', 'mean_valid_RE', 'best_valid_CE', 'best_valid_RE',
                     'mean_epochs'])
    # print(1/0)

    # loop the models setups
    for i in range(len(hyperparameters)):
        params = hyperparameters[i]
        norm_func = get_normalize_func(params[0])
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
        validation_results = np.array(pool.starmap(parallel_cross_validation,
                                                   zip(np.arange(10), repeat([split, train_inputs, train_labels, i, params]))))
        # print(validation_results)
        # print(validation_results.shape)

        # mean from results
        means = np.mean(validation_results, axis=0)
        mean_CE = means[1]
        mean_RE = means[2]
        mean_epochs = means[3]
        bests = np.min(validation_results,axis=0)
        best_CE = bests[1]
        best_RE = bests[2]

        # save hyperparameters and results in csv
        writer.writerow([i, params[0], params[1], params[2], params[3],
                         params[4], params[5], params[6], params[7], params[8],
                         params[9], params[10], params[11],
                         mean_CE, mean_RE, best_CE, best_RE, mean_epochs])



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




