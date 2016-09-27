'''
Created on Sep 26, 2016

@author: 24lee_000
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

runs = 10
print('TEST r values: 5, 1, 0.5, 0.1 0.05, 0.01, 0.005, 0.001')
if (mean_std):
    print('\tEach value is run ' + str(runs) + ' times. A mean and std dev for the accuracy is gathered.')
print('\nAGAINST\n\t a5a.train\n')

rng = range(runs)
r_vals = [5, 1,0.5,0.1,0.05,0.01, 0.005, 0.001]
vals = [0]*runs

mean_str = ''
std_str = ''
std_upd = ''
std_acc = ''

for r in r_vals:
    
    for inx in rng:
        W_b = run_perceptron('res/a5a.train', r)
        ep = evaluate_perceptron('res/a5a.train', W_b['W'], W_b['b'])
        vals[inx] = ep['accuracy']
    print('EVALUATE r =\t' + str(r))
    
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

for r in r_vals:
    
    for inx in rng:
        W_b = run_perceptron('res/a5a.train', r)
        ep = evaluate_perceptron('res/a5a.test', W_b['W'], W_b['b'])
        vals[inx] = ep['accuracy']
        
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
print('TEST mu values: 5, 1, 0.5, 0.11 0.05, 0.01')
r=1
if (mean_std):
    print('\tEach value is run ' + str(runs) + ' times. A mean and std dev for the accuracy is gathered.')
print('\tr:' + str(r))
mu_vals = [0.5, 1, 2, 3, 4, 5, 6]

print('\nAGAINST\n\t a5a.train\n')

mean_str = ''
std_str = ''
std_upd = ''
std_acc = ''

for mu in mu_vals:
    
    for inx in rng:
        W_b = run_perceptron_margin('res/a5a.train', r, mu)
        ep = evaluate_perceptron('res/a5a.train', W_b['W'], W_b['b'])
        vals[inx] = ep['accuracy']
    print('EVALUATE mu =\t' + str(mu))
  
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

for mu in mu_vals:
    
    for inx in rng:
        W_b = run_perceptron_margin('res/a5a.train', r, mu)
        ep = evaluate_perceptron('res/a5a.test', W_b['W'], W_b['b'])
        vals[inx] = ep['accuracy']
    print('EVALUATE mu =\t' + str(mu))
  
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
print('= END - 02 EXPERIMENT =')
print('=======================')