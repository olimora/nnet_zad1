import numpy as np
import random
from util import *
from classifier import *
import multiprocessing as mp
from itertools import repeat
import datetime
import csv
import pandas

def parallel_cross_validation(split_ID, passed_data):
    #passed data from common main
    split = passed_data[0]
    train_inputs = passed_data[1]
    train_labels = passed_data[2]
    model_ID = passed_data[3]
    parameters = passed_data[4]

    print('Model ', model_ID, ', Split ', split_ID, ': started')

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
    trainCEs, trainREs, validCE, validRE, epoch = model.train(estim_inputs, estim_labels, valid_inputs, valid_labels,
                                                               alpha=parameters[4],
                                                               momentum=parameters[5],
                                                               min_accuracy=parameters[6],
                                                               max_epoch=parameters[7],
                                                               min_delay_expectancy=parameters[8],
                                                               q_size=parameters[9],
                                                               raised_err_threashold=parameters[10],
                                                               acc_err_threshold=parameters[11],
                                                               trace_text=False, trace_plots=False, trace_interval=10)
    testCE, testRE = model.test(valid_inputs, valid_labels)
    print('Model ', model_ID, ', Split ', split_ID, ': stopped')
    return np.array([split_ID, testCE, testRE, epoch])

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
        ['std',  # data normalization type
         [2, 24, 12, 3], ['tanh', 'sig', 'sig'], ['uniform', [0, 1]],  # layers, functions, distribution and scale
         0.15, 0.1,  # aplha, momentum
         97, 250, 50,  # min_accuracy, max_epoch, min_delay_expectancy
         20, 0.66, 0.002])  # q_size, raised_err_threashold, acc_err_threshold
    # hyperparameters.append(
    #     ['std',  # data normalization type
    #      [2, 24, 12, 3], ['tanh', 'sig', 'sig'], ['uniform', [0, 1]],  # layers, functions, distribution and scale
    #      0.12, 0.05,  # aplha, momentum
    #      97, 250, 50,  # min_accuracy, max_epoch, min_delay_expectancy
    #      30, 0.66, 0.002])  # q_size, raised_err_threashold, acc_err_threshold
    # hyperparameters.append(
    #     ['std',  # data normalization type
    #      [2, 24, 12, 3], ['tanh', 'sig', 'sig'], ['uniform', [0, 1]],  # layers, functions, distribution and scale
    #      0.12, 0.05,  # aplha, momentum
    #      97, 250, 50,  # min_accuracy, max_epoch, min_delay_expectancy
    #      30, 0.66, 0.002])  # q_size, raised_err_threashold, acc_err_threshold




    date_time = datetime.datetime.now()
    file_name = 'D://skola//NNET//source//zadanie1_results//validation_results_add.csv' \
        .format(date_time.year, date_time.month, date_time.day, date_time.hour, date_time.minute, date_time.second)
    file = open(file_name, 'a', newline='')
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # writer.writerow(['model', 'data_normalization', 'layers', 'functions', 'distribution',
    #                  'aplha', 'momentum', 'min_accuracy', 'max_epoch', 'min_delay_expectancy',
    #                  'q_size', 'raised_err_threashold', 'acc_err_threshold',
    #                  'mean_valid_CE', 'mean_valid_RE', 'best_valid_CE', 'best_valid_RE',
    #                  'mean_epochs'])
    # print(1/0)

    # loop the models setups
    for i in range(len(hyperparameters)):
        params = hyperparameters[i]
        norm_func = get_normalize_func(params[0])
        train_inputs = train_inputs_org
        train_inputs[0] = norm_func(train_inputs[0])
        train_inputs[1] = norm_func(train_inputs[1])
        train_labels = train_labels_org

        # split to 10 sets
        ind = np.arange(len(train_labels_org))
        random.shuffle(ind)
        split = np.array(np.split(ind, 10))

        # parallel cross validation
        pool = mp.Pool(processes=4)  # removing processes argument makes the code run on all available cores
        validation_results = np.array(pool.starmap(parallel_cross_validation,
                                                   zip(np.arange(10), repeat([split, train_inputs, train_labels, i, params]))))

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

    file.close()







