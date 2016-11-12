'''
Created on Sep 24, 2016

@author: Leland Stenquist
'''
import csv
import ast

from SVM import dot_product

#
#
#
def evaluate_svm(labels_file, data_file, W, b):
    
    test_data = []
    y_vals = []
    
    wrong = 0;
    right = 0;
    
    with open(labels_file) as csvfile:
        # Read in the CSV
        labels_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

        # fill in y_vals with CSV data
        for row in labels_reader:
            y_vals.append(int(row.pop(0)))

    with open(data_file) as csvfile:
        # Read in the CSV
        data_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

        # fill in training_data with CSV data
        for row in data_reader:
            count = 0
            dict_str = '{'
            for val in row:
                dict_str = dict_str + str(count) + ":" + val + ','
                count += 1
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
    