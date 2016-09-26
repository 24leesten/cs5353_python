'''
Created on Sep 23, 2016

@author: Leland Stenquist
'''

from PerceptronAlgorithmSanity import run_perceptron

print('=========================')
print('= START - 01 EXPERIMENT =')
print('=========================')
W_b = run_perceptron()
print('Weights: ' + str(W_b['W']))
print('Bias: ' + str(W_b['b']))
print('Total Rows: ' + str(W_b['length']))
print('updates: ' + str(W_b['count']))

print('\n=======================')
print('= END - 01 EXPERIMENT =')
print('=======================')