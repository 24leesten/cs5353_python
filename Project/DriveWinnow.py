import warnings
warnings.simplefilter("ignore", RuntimeWarning)

from Scanner import scan
from Winnow import winnow, test_winnow
from Helper import limit_features, feature_scaling, standardiztion

r    = 2
epoch = 1

data = scan('res/training1.data')
test_data = data['d']
test_y = data['l']

test_data = standardiztion(test_data)
test_data=limit_features(test_data,[36, 24, 22, 42, 402, 52,  32, 29, 20, 51])

print("WINNOW")

weight_perceptron = winnow(test_data, test_y, epoch, mu)
results_perceptron = test_winnow(test_data, test_y, weight_perceptron)

print("POSITIVE: " + str(test_y.count(1)))
print("NEGATIVE: " + str(test_y.count(-1)))
print("")



print("WEIGHT: " + str(weight_perceptron))
# get the accuracy
accuracy = results_perceptron["correct"] / (results_perceptron["correct"] + results_perceptron["wrong"])

print(results_perceptron)
print(accuracy)
