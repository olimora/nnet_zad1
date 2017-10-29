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
    # 2, 20, 3 = 2 vstupne hodnoty, 20 skrytych neuronov, 3 vystupne - pre kazdu classu jeden
    model = MLPClassifier([estim_inputs.shape[0], 20, np.max(estim_labels) + 1])
    trainCEs, trainREs = model.train(estim_inputs, estim_labels, alpha=0.05, eps=50,
                                     trace=False, trace_interval=10, model_num=split_id)
    testCE, testRE = model.test(valid_inputs, valid_labels)
    print('Cross-Validation testing error: validation set = {:d}, CE = {:6.2%}, RE = {:.5f}'.format(split_id, testCE, testRE))

    return np.array([split_id, testCE, testRE])

if __name__ == '__main__':

    ## load data
    train_inputs = np.loadtxt('2d.trn.dat', skiprows=1, usecols=(0, 1)).T
    train_labels = np.loadtxt('2d.trn.dat', dtype='S20', skiprows=1, usecols=(2)).astype(str).T
    train_labels = labels_to_nums(train_labels)
    test_inputs = np.loadtxt('2d.tst.dat', skiprows=1, usecols=(0, 1)).T
    test_labels = np.loadtxt('2d.tst.dat', dtype='S20', skiprows=1, usecols=(2)).astype(str).T
    test_labels = labels_to_nums(test_labels)

    # plot_scatter("training data", test_inputs[0], test_inputs[1], test_labels)

    ## normalize
    train_inputs[0] = (train_inputs[0] - np.amin(train_inputs[0])) / (np.amax(train_inputs[0]) - np.amin(train_inputs[0]))
    train_inputs[1] = (train_inputs[1] - np.amin(train_inputs[1])) / (np.amax(train_inputs[1]) - np.amin(train_inputs[1]))
    test_inputs[0] = (test_inputs[0] - np.amin(test_inputs[0])) / (np.amax(test_inputs[0]) - np.amin(test_inputs[0]))
    test_inputs[1] = (test_inputs[1] - np.amin(test_inputs[1])) / (np.amax(test_inputs[1]) - np.amin(test_inputs[1]))


    ## cross validation ################################

    ## split to 10 sets
    # ind = np.arange(len(train_labels))
    # random.shuffle(ind)
    # split = np.array(np.split(ind, 10))

    # removing processes argument makes the code run on all available cores
    # pool = mp.Pool(processes=4)
    # results = np.array(pool.starmap(parallel_cross_validation, zip(np.arange(10), repeat([split, train_inputs, train_labels]))))
    # print(results)

    model = MLPClassifier([train_inputs.shape[0], 20, 6, np.max(train_labels) + 1],
                          ['tanh', 'sig', 'lin'], ['uniform', [0, 1]],
                          model_ID=0)
    trainCEs, trainREs = model.train(train_inputs, train_labels, test_inputs, test_labels,
                                     alpha=0.05, momentum = 0.2,
                                     min_accuracy=97, max_epoch=500, min_delay_expectancy=50,
                                     q_size=30, raised_err_threashold=0.66, acc_err_threshold=0.001,
                                     trace=False, trace_interval=10)

    testCE, testRE = model.test(test_inputs, test_labels)
    print('Final testing error: CE = {:6.2%}, RE = {:.5f}'.format(testCE, testRE))

    plot_both_errors(trainCEs, trainREs, testCE, testRE, block=False)

# ## iterate over them, pick 1 validation set and the rest for training
# best_model = None
# best_model_number = None
# best_RE = None
# best_trainCEs = None
# bect_trainREs = None
# results_table = np.zeros((10,3))
# for i in np.arange(10):
#     valid_ind = split[i]
#     train_ind = np.concatenate(split[np.delete(np.arange(10), i, axis=0)])
#     cv_train_inputs = train_inputs[:, train_ind]
#     cv_train_labels = train_labels[train_ind]
#     cv_valid_inputs = train_inputs[:, valid_ind]
#     cv_valid_labels = train_labels[valid_ind]
#
#     ## train & validate
#     cv_model = MLPClassifier(cv_train_inputs.shape[0], 20, np.max(cv_train_labels) + 1) # 2, 20, 3 = 2 vstupne hodnoty, 20 skrytych neuronov, 3 vystupne - pre kazdu classu jeden
#     cv_trainCEs, cv_trainREs = cv_model.train(cv_train_inputs, cv_train_labels, alpha=0.05, eps=500, trace=False, trace_interval=10, model_num=i)
#     cv_testCE, cv_testRE = cv_model.test(cv_valid_inputs, cv_valid_labels)
#     print('Cross-Validation testing error: CE = {:6.2%}, RE = {:.5f}'.format(cv_testCE, cv_testRE))
#     results_table[i, 0] = i
#     results_table[i, 1] = cv_testCE
#     results_table[i, 2] = cv_testRE
#
#     ## keep the best model
#     if i == 0:
#         best_model = cv_model
#         best_model_number = i
#         best_RE = cv_testRE
#         bect_trainCEs = cv_trainCEs
#         bect_trainREs = cv_trainREs
#     else:
#         if cv_testRE < best_RE: # ak ma mensiu chybu - je lepsi
#             best_model = cv_model
#             best_model_number = i
#             best_RE = cv_testRE
#             bect_trainCEs = cv_trainCEs
#             bect_trainREs = cv_trainREs
#
# ## test the best model on test data
# testCE, testRE = best_model.test(test_inputs, test_labels)
# print('Final testing: Model: {1d}, error: CE = {:6.2%}, RE = {:.5f}'.format(testCE, testRE))
#
# plot_both_errors(bect_trainCEs, bect_trainREs, testCE, testRE, block=False)
# print(results_table)


## train
# model = MLPClassifier(train_inputs.shape[0], 20, np.max(train_labels) + 1) # 2, 20, 3 = 2 vstupne hodnoty, 20 skrytych neuronov, 3 vystupne - pre kazdu classu jeden
# trainCEs, trainREs = model.train(train_inputs, train_labels, alpha=0.05, eps=500, trace=True, trace_interval=10)
#
# testCE, testRE = model.test(test_inputs, test_labels)
# print('Final testing error: CE = {:6.2%}, RE = {:.5f}'.format(testCE, testRE))
#
# plot_both_errors(trainCEs, trainREs, testCE, testRE, block=False)
