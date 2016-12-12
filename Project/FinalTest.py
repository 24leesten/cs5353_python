import warnings
warnings.simplefilter("ignore", RuntimeWarning)

from Scanner import scan
from Perceptron import perceptron, test_perceptron
from LogRegClass import gradient_descent_logistic_reg, test_log_reg_class
from SVM import svm, test_svm
from Helper import limit_features, feature_scaling, standardiztion

mu    = 0.25
epoch = 10

train = scan('res/training1.data')
train_data = train['d']
train_y = train['l']
test1 = scan('res/test.data/AHU 13.csv')
test1_data = test1['d']
test1_y = test1['l']
test2 = scan('res/test.data/AHU38 1.csv')
test2_data = test2['d']
test2_y = test2['l']
test3 = scan('res/test.data/AHU19B 1.csv')
test3_data = test3['d']
test3_y = test3['l']

train_data = standardiztion(train_data)
test1_data = standardiztion(test1_data)
test2_data = standardiztion(test2_data)
test3_data = standardiztion(test3_data)
train_data=limit_features(train_data,[36, 24, 22, 42, 402, 52,  32, 29, 20, 51])
test1_data=limit_features(test1_data,[36, 24, 22, 42, 402, 52,  32, 29, 20, 51])
test2_data=limit_features(test2_data,[36, 24, 22, 42, 402, 52,  32, 29, 20, 51])
test3_data=limit_features(test3_data,[36, 24, 22, 42, 402, 52,  32, 29, 20, 51])

print("TEST 1 Y")
print("POSITIVE: " + str(test1_y.count(1)))
print("NEGATIVE: " + str(test1_y.count(-1)))

print("TEST 2 Y")
print("POSITIVE: " + str(test2_y.count(1)))
print("NEGATIVE: " + str(test2_y.count(-1)))

print("TEST 3 Y")
print("POSITIVE: " + str(test3_y.count(1)))
print("NEGATIVE: " + str(test3_y.count(-1)))

# +++++++++++++++++++++++++++++++++++
# PERCEPTTRON
# +++++++++++++++++++++++++++++++++++
print("PERCEPTRON")

weight_perceptron = perceptron(train_data, train_y, epoch, mu)
print("WEIGHT: " + str(weight_perceptron))
results_perceptron = test_perceptron(test1_data, test1_y, weight_perceptron)
accuracy = results_perceptron["correct"] / (results_perceptron["correct"] + results_perceptron["wrong"])
print("AHU 13: " + str(accuracy))
results_perceptron = test_perceptron(test2_data, test2_y, weight_perceptron)
accuracy = results_perceptron["correct"] / (results_perceptron["correct"] + results_perceptron["wrong"])
print("AHU 38: " + str(accuracy))
results_perceptron = test_perceptron(test3_data, test3_y, weight_perceptron)
accuracy = results_perceptron["correct"] / (results_perceptron["correct"] + results_perceptron["wrong"])
print("AHU 19B: " + str(accuracy))


# +++++++++++++++++++++++++++++++++++
# SVM
# +++++++++++++++++++++++++++++++++++
print("")
print("SVM")
c = 32
epoch = 10

weigth_svm = svm(train_data, train_y, epoch, 1)
print("WEIGHT: " + str(weigth_svm))
results_svm = test_svm(test1_data, test1_y, weight_perceptron)
accuracy_svm = results_svm["correct"] / (results_svm["correct"] + results_svm["wrong"])
print("AHU 13: " + str(accuracy))
results_svm = test_svm(test2_data, test2_y, weight_perceptron)
accuracy_svm = results_svm["correct"] / (results_svm["correct"] + results_svm["wrong"])
print("AHU 38: " + str(accuracy))
results_svm = test_svm(test3_data, test3_y, weight_perceptron)
accuracy_svm = results_svm["correct"] / (results_svm["correct"] + results_svm["wrong"])
print("AHU 19B: " + str(accuracy))

# +++++++++++++++++++++++++++++++++++
# LOG REG
# +++++++++++++++++++++++++++++++++++
# print("")
# print("LOG REG")
# sigma = 16
# epoch = 100
#
# weight_log = gradient_descent_logistic_reg(train_data, train_y, epoch, sigma)
# print("WEIGHT: " + str(weight_log))
# results_log = test_log_reg_class(test1_data, test1_y, weight_log)
# accuracy_log = results_log["correct"] / (results_log["correct"] + results_log["wrong"])
# print("AHU 13: " + str(accuracy))
# results_log = test_log_reg_class(test2_data, test2_y, weight_log)
# accuracy_log = results_log["correct"] / (results_log["correct"] + results_log["wrong"])
# print("AHU 38: " + str(accuracy))
# results_log = test_log_reg_class(test3_data, test3_y, weight_log)
# accuracy_log = results_log["correct"] / (results_log["correct"] + results_log["wrong"])
# print("AHU 19B: " + str(accuracy))

