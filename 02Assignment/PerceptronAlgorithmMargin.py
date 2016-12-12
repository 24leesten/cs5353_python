'''
Created on Sep 23, 2016

@author: Leland Stenquist
'''
import csv
import ast
import random
from PerceptronAlgorithm import dot_product
from PerceptronAlgorithm import update_W
from PerceptronAlgorithm import debug_perceptron
import re

MU = 0

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
def perceptron_alg_margin(X,y,W,b,r,count):
    W_b = {'W':W,'b':b,'count':count}
    if(False):
        debug_perceptron(X,y,W,b,r)
    if(y * (dot_product(W,X) + b) < MU):
        count = count + 1
        W_b = {'W':update_W(W,r,y,X),'b':(b + (r * y)),'count':count}
        return W_b
    return W_b

# read in the training data
#
# MAIN
#
def run_perceptron_margin(file, r, mu, epochs = 1):

    global MU
    MU = mu

    # create the variables that will be used
    training_data = []
    y_vals = []
    W_b = {'W':{},'b':random.random(),'count':0}

    with open(file) as csvfile:
        # Read in the CSV
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        
    
        n = 0
        # fill in variables with CSV data
        for row in reader:
            y_vals.append(int(row.pop(0)))
            dict_str = '{'
            for val in row:
                m = re.search('(?<=:)[^\s]*', val)
                if m is not None:
                    num = m.group(0)
                    if not num.replace('.','').isdigit():
                        val = val.replace(num,'0')
                dict_str = dict_str + val + ','
            dict_str = dict_str[:-1] + '}'
            # try:
            training_data.append(ast.literal_eval(dict_str))
            # except SyntaxError:
            #     print(n)
            #     break
            n += 1

    if(True):
        print("POSITIVE: " + str(y_vals.count(1)))
        print("NEGATIVE: " + str(y_vals.count(-1)))

    
    count = 0
    range_td = list(range(len(y_vals)))
    while count < epochs:      
        if (epochs > 1):
            random.shuffle(range_td)
            
        # Loop through all the rows
        for row in range_td:
            W_b = perceptron_alg_margin(training_data[row],y_vals[row],W_b['W'],W_b['b'],r,W_b['count'])
        count = count + 1
        
    W_b['tests'] = len(y_vals) * epochs
    return(W_b)
        
        