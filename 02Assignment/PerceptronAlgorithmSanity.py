'''
Created on Sep 21, 2016

@author: Leland Stenquist
'''
import csv
import ast

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
        return 0 

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
    if(bool(W) == False):
        return product
    
    for x in X:
        w = get_weight(W, x)
        product = product + X[x]*w
    return product

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
def matrix_mult_constant(X,c):
    M={}
    for x in X:
        M[x] = X[x] * c
    return M

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
def update_W(W,r,y,X):
    M = matrix_mult_constant(X, r*y)
    W = matrix_add(W,M)
    return W

#
# debug perceptron input
#
# @Params:
#    X(dict) : dict of x values
#    y(int) : y value
#    W(dict) : dict of weights
#    b(int) : b value
#    r(int) : r value
#
def debug_perceptron(X,y,W,b,r):
    print('X: ' + str(X))
    print('y: ' + str(y))
    print('W: ' + str(W))
    print('b: ' + str(b))
    print('r: ' + str(r))

# the actual perceptron Algorithm  
#
# debug perceptron input
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
def perceptron_alg(X,y,W,b,r,count):
    W_b = {'W':W,'b':b,'count':count}
    if(False):
        debug_perceptron(X,y,W,b,r)
    if(y * (dot_product(W,X) + b) <= 0):
        count = count + 1
        W_b = {'W':update_W(W,r,y,X),'b':(b + (r * y)),'count':count}
        return W_b
    return W_b

# read in the training data
#
# MAIN
#
def run_perceptron():  
        # create the variables that will be used
    training_data = []
    y_vals = []
    W_b = {'W':{},'b':0,'count':0}
    r = 1

    with open('res/table2') as csvfile:
        # Read in the CSV
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        
    
        
        # fill in variables with CSV data
        for row in reader:
            y_vals.append(int(row.pop(0)))
            dict_str = '{'
            for val in row:
                dict_str = dict_str + val + ','
            dict_str = dict_str[:-1] + '}'
            training_data.append(ast.literal_eval(dict_str))
    
    # Loop through all the rows
    for row in range(0,len(y_vals)):
        W_b = perceptron_alg(training_data[row],y_vals[row],W_b['W'],W_b['b'],r,W_b['count'])
    W_b['length'] = len(y_vals)
    return W_b
                