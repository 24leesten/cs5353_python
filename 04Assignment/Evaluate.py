#
#
#

import csv
import ast
from SVM import dot_product


def evaluate_svm(y_vals, test_data, W, b):
    wrong = 0
    right = 0

    for row in range(len(test_data)):

        X = test_data[row]
        y = y_vals[row]

        if y * (dot_product(W, X) + b) <= 0:
            wrong = wrong + 1
        else:
            right = right + 1

    accuracy = float(float(right) / (float(right) + float(wrong)))
    error = float(float(wrong) / (float(right) + float(wrong)))
    return {'right': right, 'wrong': wrong, 'accuracy': accuracy, 'error': error}


#
#
#
def precision_svm(y_vals, test_data, W, b):

    TP = 0;
    FP = 0;
    FN = 0;

    for row in range(len(test_data)):

        X = test_data[row]
        y = y_vals[row]

        if y * (dot_product(W, X) + b) <= 0:
            if y == 1:
                FN += 1
            else:
                FP += 1
        else:
            if y == 1:
                TP += 1

    return {'FN': FN, 'FP': FP, 'TP': TP}


def print_eval(E_P):
    print('SVM EVALUATION')
    print('Right:\t' + str(E_P['right']))
    print('Wrong:\t' + str(E_P['wrong']))
    print('Accuracy:\t' + str(E_P['accuracy']))
    print('Error:\t' + str(E_P['error']))
    print('\n')


def print_prec(p):
    print('Precision Values')
    print('TP:\t' + str(p['TP']))
    print('FP:\t' + str(p['FP']))
    print('FN:\t' + str(p['FN']))
    print('\n')


def eval_prec(prec):
    tp = float(prec['TP'])
    fp = float(prec['FP'])
    fn = float(prec['FN'])

    p = 0
    r = 0
    f = 0
    if tp > 0:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f = 2 * (p * r) / (p + r)
    return {'p': p, 'r': r, 'f': f}


def print_eval_prec(p):
    print('Precision Values')
    print('Precision:\t' + str(p['p']))
    print('Recal:\t' + str(p['r']))
    print('F-Value:\t' + str(p['f']))
    print('\n')
