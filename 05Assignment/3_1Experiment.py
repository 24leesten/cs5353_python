'''
Created on Sep 23, 2016

@author: Leland Stenquist
'''
from SVM import run_svm
from SVM import scan
from Evaluate import evaluate_svm
from Evaluate import print_eval
from Evaluate import precision_svm
from Evaluate import eval_prec
from Evaluate import print_eval_prec
from Helper import avg

trials = 1
epochs = 20
C = [pow(2,-2), pow(2,-1), 2, pow(2,2), pow(2,3)]
gamma = [0.0001, 0.5, 0.9]

one = True
two = True
three = True

print("=====================================")
print("=========== Experiment 3.1 ==========")
print("=====================================")
print("")
print("Trials: " + str(trials))
print("Epochs: " + str(epochs))

hand_train = scan("res/handwriting/train.labels","res/handwriting/train.data")
hand_train_data=hand_train['d']
hand_train_label=hand_train['l']
hand_test = scan("res/handwriting/test.labels","res/handwriting/test.data")
hand_test_data=hand_test['d']
hand_test_label=hand_test['l']
madelon_train = scan("res/madelon/madelon_train.labels", "res/madelon/madelon_train.data")
madelon_train_data=madelon_train['d']
madelon_train_label=madelon_train['l']
madelon_test = scan("res/madelon/madelon_test.labels", "res/madelon/madelon_test.data")
madelon_test_data=madelon_test['d']
madelon_test_label=madelon_test['l']


print("")
print("Files Read In")
print("")

if one:
    print("=====================================")
    print("========== Experiment 3.1.1 =========")
    print("=====================================")
    print("")
    W_b = run_svm(hand_train_label,hand_train_data,epochs)
    evaluation = evaluate_svm(hand_train_label,hand_train_data,W_b['W'],W_b['b'])
    print("=========== Training Data ===========")
    print_eval(evaluation)
    evaluation = evaluate_svm(hand_test_label,hand_test_data,W_b['W'],W_b['b'])
    print("============= Test Data =============")
    print_eval(evaluation)

if two:
    print("=====================================")
    print("========== Experiment 3.1.2 =========")
    print("=====================================")
    print("")


    for c in C:
        for g in gamma:
            print("== TESTING -> C: " + str(c) + " | Gamma: " + str(g) + " ==")
            print("")
            train_acc = []
            test_acc = []
            for i in range(trials):
                W_b = run_svm(madelon_train_label, madelon_train_data, epochs,c,g)
                evaluation = evaluate_svm(madelon_train_label, madelon_train_data, W_b['W'], W_b['b'])
                train_acc.append(evaluation['accuracy'])
                evaluation = evaluate_svm(madelon_test_label, madelon_test_data, W_b['W'], W_b['b'])
                test_acc.append(evaluation['accuracy'])

            print("=========== Training Data ===========")
            print("Average Accuracy: " + str(avg(train_acc)))
            print("")
            print("============= Test Data =============")
            print("Average Accuracy: "+ str(avg(test_acc)))
            print("")

if three:
    print("=====================================")
    print("========== Experiment 3.1.3 =========")
    print("=====================================")

    print("")
    print("== 1.1 ==")
    print("")
    W_b = run_svm(hand_train_label,hand_train_data,epochs)
    precision = precision_svm(hand_train_label,hand_train_data,W_b['W'],W_b['b'])
    precision = eval_prec(precision)
    print("=========== Training Data ===========")
    print_eval_prec(precision)
    precision = precision_svm(hand_test_label,hand_test_data,W_b['W'],W_b['b'])
    precision = eval_prec(precision)
    print("============= Test Data =============")
    print_eval_prec(precision)

    print("")
    print("== 1.2 ==")
    print("")
    for c in C:
        for g in gamma:
            print("== TESTING -> C: " + str(c) + " | Gamma: " + str(g) + " ==")
            print()
            train_p = []
            train_r = []
            train_f = []
            test_p = []
            test_r = []
            test_f = []
            for i in range(trials):
                W_b = run_svm(madelon_train_label, madelon_train_data, epochs,c,g)
                precision = precision_svm(madelon_train_label, madelon_train_data, W_b['W'], W_b['b'])
                precision = eval_prec(precision)
                train_p.append(precision['p'])
                train_r.append(precision['r'])
                train_f.append(precision['f'])
                precision = precision_svm(madelon_test_label, madelon_test_data, W_b['W'], W_b['b'])
                precision = eval_prec(precision)
                test_p.append(precision['p'])
                test_r.append(precision['r'])
                test_f.append(precision['f'])

            print("=========== Training Data ===========")
            pre = {'p':avg(train_p),'r':avg(train_r),'f':avg(train_f)}
            print_eval_prec(pre)
            print("============= Test Data =============")
            pre = {'p': avg(test_p), 'r': avg(test_r), 'f': avg(test_f)}
            print_eval_prec(pre)
