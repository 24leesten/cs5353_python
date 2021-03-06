'''
Created on Sep 23, 2016

@author: Leland Stenquist
'''
import csv
import ast
import random

C = 0


def rand():
    if random.random() < 0.5:
        return random.random() * 10 * (-1)
    else:
        return random.random() * 10


#
# debug svm input
#
# @Params:
#    X(dict) : dict of x values
#    y(int) : y value
#    W(dict) : dict of weights
#    b(int) : b value
#    r(int) : r value
#
def debug_svm(X, y, W, b, gamma):
    print('X: ' + str(X))
    print('y: ' + str(y))
    print('W: ' + str(W))
    print('b: ' + str(b))
    print('gamma: ' + str(gamma))


def scan(labels_file, Attributes_file):

    # create the variables that will be used
    training_data = []
    y_vals = []

    with open(labels_file) as csvfile:
        # Read in the CSV
        labels_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

        # fill in y_vals with CSV data
        for row in labels_reader:
            if row == "":
                continue
            y_vals.append(int(row.pop(0)))

    with open(Attributes_file) as csvfile:
        # Read in the CSV
        data_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

        # fill in training_data with CSV data
        for row in data_reader:

            count = 0
            dict_str = '{'
            for val in row:
                if str(val) == "" or val is None or not str(val).isdigit:
                    continue
                dict_str = dict_str + str(count) + ":" + val + ','
                count += 1
            dict_str = dict_str[:-1] + '}'
            training_data.append(ast.literal_eval(dict_str))

    return {'d': training_data, 'l': y_vals}

#
# Get the dot product of the two Dicts that represent Vectors.   
#
# @Params:
#    W(dict) : dict of weights
#    X(dict) : dict of x values
# @Return: 
#    product(int) : dot product of the vectors
#
def dot_product(W, X):
    product = 0;
    if bool(W) == False:
        return product

    for x in X:
        w = get_weight(W, x)
        product = product + X[x] * w
    return product


#
# update W using this algorithm W = W + ryX
#
# @Params:
#    W(dict) : dict of weights
#    r(int) : r value
#    y(int) : y value
#    X(dict) : dict of x values
# @Return: 
#    product(int) : dot product of the vectors
#
def update_W(W, gamma, y, X):
    W = matrix_mult_constant(W, (1 - gamma))
    X = matrix_mult_constant(X, (gamma * C * y))
    R = matrix_add(W, X)
    return R

#
# add two matrices
#
# @Params:
#    W(dict) : dict of weights
#    X(dict) : dict of x values
# @Return: 
#    sum(dix) : sum of the two matrices
#
def matrix_add(W, X):
    for x in X:
        w = get_weight(W, x)
        W[x] = w + X[x]
    return W


#
# multiply a matrix by a constant
#
# @Params:
#    W(dict) : dict of x values
#    c(int) : constand value
# @Return: 
#    product(int) : dot product of the vectors
#
def matrix_mult_constant(X, c):
    M = {}
    for x in X:
        M[x] = X[x] * c
    return M


#
# Get the weight at the given index. If the index 
#    does not exist for that weight then it is
#    zero.
#
# @Params:
#    W(dict) : dict of weights
#    index(int) : index of the wight needed 
# @Return: 
#    weight(int)
#
def get_weight(W, index):
    if index in W.keys():
        return W[index]
    else:
        W[index] = 0;
        return W[index]


# the actual svm Algorithm
#
# debug svm input
#
# @Params:
#    X(dict) : dict of x values
#    y(int) : y value
#    W(dict) : dict of weights
#    b(int) : b value
#    r(int) : r value
# @Return
#    W_b(dict) : dict of return values
#        ['W'](dict) : dict of weights
#        ['b'](int) : b valuse
#        ['count'](int) : count of updates
# 
def svm_alg(X, y, W, b, count, gamma):
    W_b = {'W': W, 'b': b, 'count': count}
    if False:
        debug_svm(X, y, W, b, gamma)
    # SVM
    if y * (dot_product(W, X) + b) <= 1:
        count = count + 1
        W_b = {'W': update_W(W, gamma, y, X), 'b': (((1-gamma)*b)+(gamma*C*y)), 'count': count}
        return W_b
    else:
        W_b = {'W': matrix_mult_constant(W, (1 - gamma)), 'b': ((1-gamma)*b), 'count': count}
        return W_b


# read in the training data
#
# MAIN
#
def run_svm(y_vals, training_data, epochs=-1, c=1, gamma=0.01, bias=0):
    global C
    C = c

    W_b = {'W': {}, 'b': bias, 'count': 0}



    count = 0
    range_td = list(range(len(training_data[0])))

    if epochs == -1:
        epochs = len(y_vals)
    while count < epochs:
        random.shuffle(range_td)
        # Loop through all the rows
        for row in range_td:
            W_b = svm_alg(training_data[row], y_vals[row], W_b['W'], W_b['b'], W_b['count'], gamma)

            gamma = (gamma / (1 + gamma * (count / C)))
        count += 1

    W_b['tests'] = len(y_vals) * epochs
    return (W_b)



