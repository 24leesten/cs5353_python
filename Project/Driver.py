import warnings
warnings.simplefilter("ignore", RuntimeWarning)

from Scanner import scan
from Perceptron import perceptron, test_perceptron
from LogRegClass import gradient_descent_logistic_reg, test_log_reg_class
from SVM import svm, test_svm

mu    = 5
epoch = 10

data = scan('res/AHU27.csv')
test_data = data['d']
test_y = data['l']

weight_perceptron = perceptron(test_data, test_y, epoch, mu, 0)
results_perceptron = test_perceptron(test_data, test_y, weight_perceptron)

print("PERCEPTRON")
print("POSITIVE: " + str(test_y.count(1)))
print("NEGATIVE: " + str(test_y.count(-1)))

# get the accuracy
accuracy = results_perceptron["correct"] / (results_perceptron["correct"] + results_perceptron["wrong"])

print(results_perceptron)
print(accuracy)

sigma = 16
epoch = 100

weight_log = gradient_descent_logistic_reg(test_data, test_y, epoch, sigma, 0, 0.01)
results_log = test_log_reg_class(test_data, test_y, weight_log)

print("")

print("LOG REG")

# get the accuracy
accuracy_log = results_log["correct"] / (results_log["correct"] + results_log["wrong"])

print(results_log)
print(accuracy_log)

weigth_svm = svm(test_data, test_y, epoch)
results_svm = test_svm(test_data, test_y, weigth_svm)

print("")

print("SVM")

# get the accuracy
accuracy_svm = results_svm["correct"] / (results_svm["correct"] + results_svm["wrong"])

print(results_svm)
print(accuracy_svm)

print("")
print("WEIGHTS")
print(weight_perceptron[0])
print(weight_log[0])
print(weigth_svm[0])