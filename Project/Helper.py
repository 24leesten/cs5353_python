from LogRegClass import gradient_descent_logistic_reg, test_log_reg_class
from Perceptron import perceptron, test_perceptron
from Winnow import winnow, test_winnow
from SVM import svm,test_svm
from Scanner import scan
import numpy as np


def cross_validation_scan():
    # v1_train = scan("res/x_validation/v1.train")
    # v1_test = scan("res/x_validation/v1.test")
    # v2_train = scan("res/x_validation/v2.train")
    # v2_test = scan("res/x_validation/v2.test")
    # v3_train = scan("res/x_validation/v3.train")
    # v3_test = scan("res/x_validation/v3.test")
    # v4_train = scan("res/x_validation/v4.train")
    # v4_test = scan("res/x_validation/v4.test")
    # v5_train = scan("res/x_validation/v5.train")
    # v5_test = scan("res/x_validation/v5.test")
    # v6_train = scan("res/x_validation/v6.train")
    # v6_test = scan("res/x_validation/v6.test")

    v1_train = scan("res/xv/f1.train")
    v1_test = scan("res/xv/f1.xv")
    v2_train = scan("res/xv/f2.train")
    v2_test = scan("res/xv/f2.xv")
    v3_train = scan("res/xv/f3.train")
    v3_test = scan("res/xv/f3.xv")
    v4_train = scan("res/xv/f4.train")
    v4_test = scan("res/xv/f4.xv")
    v5_train = scan("res/xv/f5.train")
    v5_test = scan("res/xv/f5.xv")
    v6_train = scan("res/xv/f6.train")
    v6_test = scan("res/xv/f6.xv")

    return {"v1": [v1_train, v1_test], "v2": [v2_train, v2_test], "v3": [v3_train, v3_test],
            "v4": [v4_train, v4_test], "v5": [v5_train, v5_test], "v6": [v6_train, v6_test]}


def cross_validation_log(files, epochs, delta, bias=1, r=0.01):
    accuracy = []

    for file in files:
        train_file = files[file][0]
        test_file = files[file][1]

        training_set = train_file['d']
        training_set = feature_scaling(training_set)
        training_set = limit_features(training_set, [36, 24, 22, 42, 402, 52, 32, 29, 20, 51])
        training_lbl = train_file['l']
        test_set = test_file['d']
        test_set = feature_scaling(test_set)
        test_set = limit_features(test_set, [36, 24, 22, 42, 402, 52, 32, 29, 20, 51])
        test_lbl = test_file['l']

        w = gradient_descent_logistic_reg(training_set, training_lbl, epochs, delta, bias, r)
        print(w)
        c = test_log_reg_class(test_set, test_lbl, w)
        accuracy.append(c["correct"] / (c["correct"] + c["wrong"]))

    return sum(accuracy) / len(accuracy)


def cross_validation_per(files, epochs, mu, bias=1, r=0.01):
    accuracy = []

    for file in files:
        train_file = files[file][0]
        test_file = files[file][1]

        training_set = train_file['d']
        training_set = limit_features(training_set, [36, 24, 22, 42, 402, 52, 32, 29, 20, 51])
        training_lbl = train_file['l']
        test_set = test_file['d']
        test_set = limit_features(test_set, [36, 24, 22, 42, 402, 52, 32, 29, 20, 51])
        test_lbl = test_file['l']

        weight_perceptron = perceptron(training_set, training_lbl, epochs, mu, bias, r)
        c = test_perceptron(test_set, test_lbl, weight_perceptron)
        accuracy.append(c["correct"] / (c["correct"] + c["wrong"]))

    return sum(accuracy) / len(accuracy)


def cross_validation_svm(files, epochs, c, bias=1, r=0.01):
    accuracy = []

    for file in files:
        train_file = files[file][0]
        test_file = files[file][1]

        training_set = train_file['d']
        training_set = limit_features(training_set, [36, 24, 22, 42, 402, 52, 32, 29, 20, 51])
        training_lbl = train_file['l']
        test_set = test_file['d']
        test_set = limit_features(test_set, [36, 24, 22, 42, 402, 52, 32, 29, 20, 51])
        test_lbl = test_file['l']

        weigth_svm = svm(training_set, training_lbl, epochs, c, bias, r)
        acc = test_svm(test_set, test_lbl, weigth_svm)
        accuracy.append(acc["correct"] / (acc["correct"] + acc["wrong"]))

    return sum(accuracy) / len(accuracy)


def cross_validation_win(files, epochs, r=2, bias=1):
    accuracy = []

    for file in files:
        train_file = files[file][0]
        test_file = files[file][1]

        training_set = train_file['d']
        # training_set = limit_features(training_set, [36, 24, 22, 42, 402, 52, 32, 29, 20, 51])
        training_lbl = train_file['l']
        test_set = test_file['d']
        # test_set = limit_features(test_set, [36, 24, 22, 42, 402, 52, 32, 29, 20, 51])
        test_lbl = test_file['l']

        weigth_svm = winnow(training_set, training_lbl, epochs, r, bias)
        acc = test_winnow(test_set, test_lbl, weigth_svm)
        accuracy.append(acc["correct"] / (acc["correct"] + acc["wrong"]))

    return sum(accuracy) / len(accuracy)

def limit_features(data, features):
    return_data = []

    for row in data:
        new_row = []
        for feature in features:
            if len(row) > feature:
                new_row.append(row[feature])
            else:
                new_row.append(0)
        return_data.append(new_row)

    return return_data

def feature_scaling(x):
    return (x-np.min(x))/(np.max(x) - np.min(x))

def standardiztion(x):
    return (x-np.mean(x))/np.std(x)

def binarize(x):
    mean = np.mean(x)