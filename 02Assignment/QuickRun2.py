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

mean_str = ''
std_str = ''
std_upd = ''
std_acc = ''

mean_std = True

r = 1
mu = 5
runs = 10
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
    
    mean_str = mean_str + '& ' + '{0:.3f}'.format(numpy.mean(vals))
    std_str = std_str + '& ' + '{0:.3f}'.format(numpy.std(vals))
    std_upd = std_upd + '& ' + str(ep['wrong'])
    std_acc = std_acc + '& ' + '{0:.3f}'.format(ep['accuracy'] * 100)
    vals = [0]*runs

print('MEAN: ' + mean_str)
print('STD: ' + std_str)
print('UPDATE: ' + std_upd)
print('ACC: ' + std_acc)
    
print('\nAGAINST\n\t a5a.test\n')

mean_str = ''
std_str = ''
std_upd = ''
std_acc = ''

for epoch in epoch_vals:
    
    for inx in rng:
        W_b = run_perceptron('res/a5a.train', r, epoch)
        ep = evaluate_perceptron('res/a5a.test', W_b['W'], W_b['b'])
        vals[inx] = ep['accuracy']
    print('EVALUATE epoch =\t' + str(epoch))

    mean_str = mean_str + '& ' + '{0:.3f}'.format(numpy.mean(vals))
    std_str = std_str + '& ' + '{0:.3f}'.format(numpy.std(vals))
    std_upd = std_upd + '& ' + str(ep['wrong'])
    std_acc = std_acc + '& ' + '{0:.3f}'.format(ep['accuracy'] * 100)
    vals = [0]*runs

print('MEAN: ' + mean_str)
print('STD: ' + std_str)
print('UPDATE: ' + std_upd)
print('ACC: ' + std_acc)


print('\nPERCEPTRON MARGIN')
print('_________________\n')
print('TEST epoch values: 3, 5')

print('\nAGAINST\n\t a5a.train\n')

mean_str = ''
std_str = ''
std_upd = ''
std_acc = ''

for epoch in epoch_vals:
    
    for inx in rng:
        W_b = run_perceptron_margin('res/a5a.train', r, mu, epoch)
        ep = evaluate_perceptron('res/a5a.train', W_b['W'], W_b['b'])
        vals[inx] = ep['accuracy']
    print('EVALUATE epoch =\t' + str(epoch))

    mean_str = mean_str + '& ' + '{0:.3f}'.format(numpy.mean(vals))
    std_str = std_str + '& ' + '{0:.3f}'.format(numpy.std(vals))
    std_upd = std_upd + '& ' + str(ep['wrong'])
    std_acc = std_acc + '& ' + '{0:.3f}'.format(ep['accuracy'] * 100)
    vals = [0]*runs

print('MEAN: ' + mean_str)
print('STD: ' + std_str)
print('UPDATE: ' + std_upd)
print('ACC: ' + std_acc)
    
print('\nAGAINST\n\t a5a.test\n')

mean_str = ''
std_str = ''
std_upd = ''
std_acc = ''

for epoch in epoch_vals:
    
    for inx in rng:
        W_b = run_perceptron_margin('res/a5a.train', r, mu, epoch)
        ep = evaluate_perceptron('res/a5a.test', W_b['W'], W_b['b'])
        vals[inx] = ep['accuracy']
    print('EVALUATE epoch =\t' + str(epoch))

    mean_str = mean_str + '& ' + '{0:.3f}'.format(numpy.mean(vals))
    std_str = std_str + '& ' + '{0:.3f}'.format(numpy.std(vals))
    std_upd = std_upd + '& ' + str(ep['wrong'])
    std_acc = std_acc + '& ' + '{0:.3f}'.format(ep['accuracy'] * 100)
    vals = [0]*runs

print('MEAN: ' + mean_str)
print('STD: ' + std_str)
print('UPDATE: ' + std_upd)
print('ACC: ' + std_acc)
    
print('\n=======================')
print('= END - 03 EXPERIMENT =')
print('=======================')