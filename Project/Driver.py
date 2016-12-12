import warnings
warnings.simplefilter("ignore", RuntimeWarning)

from Scanner import scan
from ID3 import id3, test_id3
from Perceptron import perceptron, test_perceptron
from LogRegClass import gradient_descent_logistic_reg, test_log_reg_class
from SVM import svm, test_svm
from Helper import limit_features, feature_scaling, standardiztion

# mu    = 0.25
# epoch = 1

data = scan('res/training1.data')
test_data = data['d']
test_y = data['l']

test_data = standardiztion(test_data)
test_data=limit_features(test_data,[36, 24, 22, 42, 402, 52,  32, 29, 20, 51])

# weight_perceptron = perceptron(test_data, test_y, epoch, mu)
# results_perceptron = test_perceptron(test_data, test_y, weight_perceptron)
#
# print("POSITIVE: " + str(test_y.count(1)))
# print("NEGATIVE: " + str(test_y.count(-1)))
# print("")
#
# print("PERCEPTRON")
#
# print("WEIGHT: " + str(weight_perceptron))
# # get the accuracy
# accuracy = results_perceptron["correct"] / (results_perceptron["correct"] + results_perceptron["wrong"])
#
# print(results_perceptron)
# print(accuracy)
#
# sigma = 16
# epoch = 100
#
# weight_log = gradient_descent_logistic_reg(test_data, test_y, epoch, sigma)
# results_log = test_log_reg_class(test_data, test_y, weight_log)
#
# print("")
#
# print("LOG REG")
# print("WEIGHT: " + str(weight_log))
#
# # get the accuracy
# accuracy_log = results_log["correct"] / (results_log["correct"] + results_log["wrong"])
#
# print(results_log)
# print(accuracy_log)
#
# c = 32
# epoch = 10
#
# weigth_svm = svm(test_data, test_y, epoch, 1)
# results_svm = test_svm(test_data, test_y, weigth_svm)
#
# print("")
#
# print("SVM")
# print("WEIGHT: " + str(weigth_svm))
#
# # get the accuracy
# accuracy_svm = results_svm["correct"] / (results_svm["correct"] + results_svm["wrong"])
#
# print(results_svm)
# print(accuracy_svm)
#
# print("")
# print("WEIGHTS")

print("ID3")
tree_id3 = id3(test_data, test_y, range(len(test_data[0])),  2)
result_id3 = test_id3(tree_id3,test_data,test_y)
print(result_id3)