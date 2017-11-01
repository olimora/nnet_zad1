import numpy as np
from util import *
from classifier import *
import multiprocessing as mp
from sklearn.metrics import confusion_matrix

# read csv, get best hyperparameters, create model like that and test on test data
# results = np.genfromtxt('validation_results_2017_10_30__15_ 4_21.csv', dtype=None, skip_header=1)
# df = pandas.DataFrame.from_csv('validation_results_2017_10_30__15_ 4_21.csv', sep=',')
# results = df.values # the array you are interested in
# print(type(results))

params = list()
params.append(
    ['std',  # data normalization type
     [2, 24, 12, 3], ['tanh', 'sig', 'sig'], ['uniform', [0, 1]],  # layers, functions, distribution and scale
     0.12, 0.05,  # aplha, momentum
     97, 343, 140,  # min_accuracy, max_epoch, min_delay_expectancy
     30, 0.66, 0.003])  # q_size, raised_err_threashold, acc_err_threshold

parameters = params[0]

## load data
train_inputs_org = np.loadtxt('2d.trn.dat', skiprows=1, usecols=(0, 1)).T
train_labels_org = np.loadtxt('2d.trn.dat', dtype='S20', skiprows=1, usecols=(2)).astype(str).T
train_labels_org = labels_to_nums(train_labels_org)
test_inputs_org = np.loadtxt('2d.tst.dat', skiprows=1, usecols=(0, 1)).T
test_labels_org = np.loadtxt('2d.tst.dat', dtype='S20', skiprows=1, usecols=(2)).astype(str).T
test_labels_org = labels_to_nums(test_labels_org)

# normalize data with method from parameters
norm_func = get_normalize_func(parameters[0])
train_inputs = train_inputs_org
train_inputs[0] = norm_func(train_inputs[0])
train_inputs[1] = norm_func(train_inputs[1])
train_labels = train_labels_org
test_inputs = test_inputs_org
test_inputs[0] = norm_func(test_inputs[0])
test_inputs[1] = norm_func(test_inputs[1])
test_labels = test_labels_org

# create model and train
model = MLPClassifier(parameters[1], parameters[2], parameters[3],
                      model_ID=0, split_ID=0)
trainCEs, trainREs, validCE, validRE, epochs = model.train(train_inputs, train_labels, None, None,
                                                           alpha=parameters[4],
                                                           momentum=parameters[5],
                                                           min_accuracy=0,
                                                           max_epoch=parameters[7],#mean from csv,
                                                           min_delay_expectancy=100000,
                                                           q_size=2,
                                                           raised_err_threashold=100,
                                                           acc_err_threshold=100,
                                                           trace_text=True, trace_plots=True, trace_interval=10)

# test and write to csv
testCE, testRE, predicted = model.test2(test_inputs, test_labels)
print('Final testing error: CE = {:6.2%}, RE = {:.5f}'.format(testCE, testRE))

plot_both_errors(trainCEs, trainREs, testCE, testRE, block=False)

# calculate a confusion matrix.
kung_fu_mat = confusion_matrix(test_labels, predicted)
print(kung_fu_mat)



# print(np.min(test_labels))