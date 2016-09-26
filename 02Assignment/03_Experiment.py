'''
Created on Sep 23, 2016

@author: Leland Stenquist
'''

from PerceptronAlgorithm import run_perceptron
from PerceptronAlgorithmMargin import run_perceptron_margin
from Evaluate_Perceptron import evaluate_perceptron
import numpy

print('=========================')
print('= START - 03 EXPERIMENT =')
print('=========================')
print('\nNORMAL PERCEPTRON')
print('_______________\n')

mean_std = False

r = 1
mu = 5
runs = 1
print('TEST epoch values: 3, 5')
if (mean_std):
    print('\tEach value is run ' + str(runs) + ' times. A mean and std dev for the accuracy is gathered.')
print('\tr:\t' + str(r))
print('\tmu:\t' + str(mu))
print('\nAGAINST\n\t a5a.train\n')

rng = range(runs)
epoch_vals = [3, 5]
vals = [0]*runs

for epoch in epoch_vals:
    
    for inx in rng:
        W_b = run_perceptron('res/a5a.train', r, epoch)
        ep = evaluate_perceptron('res/a5a.train', W_b['W'], W_b['b'])
        vals[inx] = ep['accuracy']
    print('EVALUATE epoch =\t' + str(epoch))
    if (mean_std):
        print('Mean Accuracy:\t\t' + str(numpy.mean(vals)))
        print('Std Deviation:\t\t' + str(numpy.std(vals)))
    print('Updates:\t\t' + str(ep['wrong']))
    print('Total Runs:\t\t' + str(ep['wrong']+ep['right']))
    print('Accuracy:\t\t' + str(ep['accuracy']))
    vals = [0]*runs
    
print('\nAGAINST\n\t a5a.test\n')

for epoch in epoch_vals:
    
    for inx in rng:
        W_b = run_perceptron('res/a5a.train', r, epoch)
        ep = evaluate_perceptron('res/a5a.test', W_b['W'], W_b['b'])
        vals[inx] = ep['accuracy']
    print('EVALUATE epoch =\t' + str(epoch))
    if (mean_std):
        print('Mean Accuracy:\t\t' + str(numpy.mean(vals)))
        print('Std Deviation:\t\t' + str(numpy.std(vals)))
    print('Updates:\t\t' + str(ep['wrong']))
    print('Total Runs:\t\t' + str(ep['wrong']+ep['right']))
    print('Accuracy:\t\t' + str(ep['accuracy']))
    vals = [0]*runs


print('\nPERCEPTRON MARGIN')
print('_________________\n')
print('TEST epoch values: 3, 5')

print('\nAGAINST\n\t a5a.train\n')

for epoch in epoch_vals:
    
    for inx in rng:
        W_b = run_perceptron_margin('res/a5a.train', r, mu, epoch)
        ep = evaluate_perceptron('res/a5a.train', W_b['W'], W_b['b'])
        vals[inx] = ep['accuracy']
    print('EVALUATE epoch =\t' + str(epoch))
    if (mean_std):
        print('Mean Accuracy:\t\t' + str(numpy.mean(vals)))
        print('Std Deviation:\t\t' + str(numpy.std(vals)))
    print('Updates:\t\t' + str(ep['wrong']))
    print('Total Runs:\t\t' + str(ep['wrong']+ep['right']))
    print('Accuracy:\t\t' + str(ep['accuracy']))
    vals = [0]*runs
    
print('\nAGAINST\n\t a5a.test\n')

for epoch in epoch_vals:
    
    for inx in rng:
        W_b = run_perceptron_margin('res/a5a.train', r, mu, epoch)
        ep = evaluate_perceptron('res/a5a.test', W_b['W'], W_b['b'])
        vals[inx] = ep['accuracy']
    print('EVALUATE epoch =\t' + str(epoch))
    if (mean_std):
        print('Mean Accuracy:\t\t' + str(numpy.mean(vals)))
        print('Std Deviation:\t\t' + str(numpy.std(vals)))
    print('Updates:\t\t' + str(ep['wrong']))
    print('Total Runs:\t\t' + str(ep['wrong']+ep['right']))
    print('Accuracy:\t\t' + str(ep['accuracy']))
    vals = [0]*runs
    
print('\n=======================')
print('= END - 03 EXPERIMENT =')
print('=======================')