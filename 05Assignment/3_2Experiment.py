'''
Created on Sep 23, 2016

@author: Leland Stenquist
'''

from ID3 import scan, id3
from ID3_Helper import get_data, ensemble_data
from SVM import run_svm
from Evaluate import evaluate_svm,print_eval,eval_prec,print_eval_prec,precision_svm
import random

epochs = 20

print("=====================================")
print("=========== Experiment 3.2 ==========")
print("=====================================")
print("")
print("Epochs: " + str(epochs))

hand_train = scan("res/handwriting/train.labels","res/handwriting/train.data")
hand_train_data=hand_train['d']
hand_train_label=hand_train['l']
hand_train_attr=hand_train['a']
hand_test = scan("res/handwriting/test.labels","res/handwriting/test.data")
hand_test_data=hand_test['d']
hand_test_label=hand_test['l']
madelon_train = scan("res/madelon/madelon_train.labels", "res/madelon/madelon_train.data",5)
madelon_train_data=madelon_train['d']
madelon_train_label=madelon_train['l']
madelon_train_attr=madelon_train['a']
madelon_test = scan("res/madelon/madelon_test.labels", "res/madelon/madelon_test.data",5)
madelon_test_data=madelon_test['d']
madelon_test_label=madelon_test['l']

print("")
print("Files Read In")
print("")

N = 5

one = True
two = True
three = True


if one:
    print("=====================================")
    print("========== Experiment 3.2.1 =========")
    print("=====================================")
    print("")
    trees = []
    for i in range(N):
        e = ensemble_data(hand_train_data,hand_train_label)
        random.shuffle(hand_train_attr)
        t = id3(e['d'], e['l'], hand_train_attr[:8])
        trees.append(t)

    data_set = get_data(trees,hand_train_data)
    W_b = run_svm(hand_train_label, data_set, epochs, 2, 0.001)
    evaluation = evaluate_svm(hand_train_label, hand_train_data, W_b['W'], W_b['b'])
    print("=========== Training Data ===========")
    print_eval(evaluation)
    evaluation = evaluate_svm(hand_test_label, hand_test_data, W_b['W'], W_b['b'])
    print("============= Test Data =============")
    print_eval(evaluation)

if two:
    print("=====================================")
    print("========== Experiment 3.2.2 =========")
    print("=====================================")
    print("")
    N=[10,30,100]
    print("== a ==")
    print("")
    for n in N:
        print("N: " + str(n))
        print("")
        trees = []
        for i in range(n):
            e = ensemble_data(madelon_train_data, madelon_train_label)
            random.shuffle(hand_train_attr)
            t = id3(e['d'], e['l'], madelon_train_attr[:11])
            trees.append(t)

        data_set = get_data(trees, madelon_train_data)
        W_b = run_svm(madelon_train_label, data_set, epochs, 2, 0.001)
        evaluation = evaluate_svm(madelon_train_label, madelon_train_data, W_b['W'], W_b['b'])
        print("=========== Training Data ===========")
        print_eval(evaluation)
        evaluation = evaluate_svm(madelon_test_label, madelon_test_data, W_b['W'], W_b['b'])
        print("============= Test Data =============")
        print_eval(evaluation)

if three:
    print("")
    print("== b ==")
    print("")

    trees = []
    for i in range(30):
        e = ensemble_data(madelon_train_data, madelon_train_label)
        t = id3(e['d'], e['l'], madelon_train_attr[:11])
        trees.append(t)

    data_set = get_data(trees, madelon_train_data)
    W_b = run_svm(madelon_train_label, data_set, epochs, 2, 0.001)
    precision = precision_svm(madelon_train_label, madelon_train_data, W_b['W'], W_b['b'])
    evaluation = evaluate_svm(madelon_train_label, madelon_train_data, W_b['W'], W_b['b'])
    precision = eval_prec(precision)
    print("=========== Training Data ===========")
    print_eval_prec(precision)
    print_eval(evaluation)
    precision = precision_svm(madelon_test_label, madelon_test_data, W_b['W'], W_b['b'])
    evaluation = evaluate_svm(madelon_test_label, madelon_test_data, W_b['W'], W_b['b'])
    precision = eval_prec(precision)
    print("============= Test Data =============")
    print_eval_prec(precision)
    print_eval(evaluation)