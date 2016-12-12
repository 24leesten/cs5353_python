'''
Created on Sep 23, 2016

@author: Leland Stenquist
'''

from PerceptronAlgorithm import run_perceptron
from PerceptronAlgorithmMargin import run_perceptron_margin
from Evaluate_Perceptron import evaluate_perceptron
import numpy

print('=========================')
print('= START - 02 EXPERIMENT =')
print('=========================')
print('\nNORMAL PERCEPTRON')
print('_______________\n')

mean_std = False

runs = 1
print('TEST r values: 5, 1, 0.5, 0.1 0.05, 0.01, 0.005, 0.001')
if (mean_std):
    print('\tEach value is run ' + str(runs) + ' times. A mean and std dev for the accuracy is gathered.')
print('\nAGAINST\n\t a5a.train\n')

rng = range(runs)
r_vals = [5, 1,0.5,0.1,0.05,0.01, 0.005, 0.001]
vals = [0]*runs

for r in r_vals:
    
    for inx in rng:
        W_b = run_perceptron('res/a5a.train', r)
        ep = evaluate_perceptron('res/a5a.train', W_b['W'], W_b['b'])
        vals[inx] = ep['accuracy']
    print('EVALUATE r =\t' + str(r))
    if (mean_std):
        print('Mean Accuracy:\t\t' + str(numpy.mean(vals)))
        print('Std Deviation:\t\t' + str(numpy.std(vals)))
    print('Updates:\t\t' + str(ep['wrong']))
    print('Total Rows:\t\t' + str(ep['wrong']+ep['right']))
    print('Accuracy:\t\t' + str(ep['accuracy']))
    vals = [0]*runs
    
print('\nAGAINST\n\t a5a.test\n')

for r in r_vals:
    
    for inx in rng:
        W_b = run_perceptron('res/a5a.train', r)
        ep = evaluate_perceptron('res/a5a.test', W_b['W'], W_b['b'])
        vals[inx] = ep['accuracy']
    print('EVALUATE r =\t' + str(r))
    if (mean_std):
        print('Mean Accuracy:\t\t' + str(numpy.mean(vals)))
        print('Std Deviation:\t\t' + str(numpy.std(vals)))
    print('Updates:\t\t' + str(ep['wrong']))
    print('Total Rows:\t\t' + str(ep['wrong']+ep['right']))
    print('Accuracy:\t\t' + str(ep['accuracy']))
    vals = [0]*runs


print('\nPERCEPTRON MARGIN')
print('_________________\n')
print('TEST mu values: 5, 1, 0.5, 0.11 0.05, 0.01')
r=1
if (mean_std):
    print('\tEach value is run ' + str(runs) + ' times. A mean and std dev for the accuracy is gathered.')
print('\tr:' + str(r))
mu_vals = [0.5, 1, 2, 3, 4, 5, 6]

print('\nAGAINST\n\t a5a.train\n')

for mu in mu_vals:
    
    for inx in rng:
        W_b = run_perceptron_margin('res/a5a.train', r, mu)
        ep = evaluate_perceptron('res/a5a.train', W_b['W'], W_b['b'])
        vals[inx] = ep['accuracy']
    print('EVALUATE mu =\t' + str(mu))
    if (mean_std):
        print('Mean Accuracy:\t\t' + str(numpy.mean(vals)))
        print('Std Deviation:\t\t' + str(numpy.std(vals)))
    print('Updates:\t\t' + str(ep['wrong']))
    print('Total Rows:\t\t' + str(ep['wrong']+ep['right']))
    print('Accuracy:\t\t' + str(ep['accuracy']))
    vals = [0]*runs
    
print('\nAGAINST\n\t a5a.test\n')

for mu in mu_vals:
    
    for inx in rng:
        W_b = run_perceptron_margin('res/a5a.train', r, mu)
        ep = evaluate_perceptron('res/a5a.test', W_b['W'], W_b['b'])
        vals[inx] = ep['accuracy']
    print('EVALUATE mu =\t' + str(mu))
    if (mean_std):
        print('Mean Accuracy:\t\t' + str(numpy.mean(vals)))
        print('Std Deviation:\t\t' + str(numpy.std(vals)))
    print('Updates:\t\t' + str(ep['wrong']))
    print('Total Rows:\t\t' + str(ep['wrong']+ep['right']))
    print('Accuracy:\t\t' + str(ep['accuracy']))
    vals = [0]*runs
    
print('\n=======================')
print('= END - 02 EXPERIMENT =')
print('=======================')