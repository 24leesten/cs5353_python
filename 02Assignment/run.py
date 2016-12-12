from Evaluate_Perceptron import evaluate_perceptron
from PerceptronAlgorithmMargin import run_perceptron_margin
r     = 1
mu    = 5
epoch = 10
W_b = run_perceptron_margin('res/AHU27.csv', r, mu, epoch)
ep = evaluate_perceptron('res/AHU27.csv', W_b['W'], W_b['b'])
print(W_b['W'])
print(ep)
