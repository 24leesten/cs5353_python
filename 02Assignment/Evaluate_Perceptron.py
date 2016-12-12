'''
Created on Sep 24, 2016

@author: Leland Stenquist
'''
import csv
import ast
from PerceptronAlgorithm import dot_product
from PerceptronAlgorithm import get_weight

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
#
#
def evaluate_perceptron(file, W, b):
    
    test_data = []
    y_vals = []
    
    wrong = 0;
    right = 0;
    
    with open(file) as csvfile:
        # Read in the CSV
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        
        # fill in variables with CSV data
        for row in reader:
            y_vals.append(int(row.pop(0)))
            dict_str = '{'
            for val in row:
                dict_str = dict_str + val + ','
            dict_str = dict_str[:-1] + '}'
            test_data.append(ast.literal_eval(dict_str))
            
    for row in range(len(test_data)):
        
        X = test_data[row]
        y = y_vals[row]
         
        if(y * (dot_product(W,X) + b) <= 0):
            wrong = wrong + 1
        else:
            right = right + 1
            
    accuracy = float(float(right)/(float(right) + float(wrong)))
    error = float(float(wrong)/(float(right) + float(wrong)))
    return {'right':right, 'wrong':wrong, 'accuracy':accuracy, 'error':error}

def print_ep(E_P):
    print('PERCEPTRON EVALUATION')
    print('Right:\t' + str(E_P['right']))
    print('Wrong:\t' + str(E_P['wrong']))
    print('Accuracy:\t' + str(E_P['accuracy']))
    print('Error:\t' + str(E_P['error']))
    print('\n')
    