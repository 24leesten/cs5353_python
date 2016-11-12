from SVM import run_svm
from Evaluate_SVM import evaluate_svm
from Evaluate_SVM import print_svm
W_b = run_svm("res/handwriting/train.labels","res/handwriting/train.data")
evaluation = evaluate_svm("res/handwriting/train.labels","res/handwriting/train.data",W_b['W'],W_b['b'])
print_svm(evaluation)