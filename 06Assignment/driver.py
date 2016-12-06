import warnings
warnings.simplefilter("ignore", RuntimeWarning)
from Scanner import scan
from LogRegClass import gradient_descent_logistic_reg, test_weight

epochs = 30
sigma = 32

#{'d': training_data, 'l': y_vals}
train=scan("res/a5a.train")
test=scan("res/a5a.test")

# get the training info
train_data = train['d']
train_labels = train['l']

# get the test info
test_data = test['d']
test_labels = test['l']

w = gradient_descent_logistic_reg(train_data, train_labels, epochs, sigma)

# parse w
weights = w['w']
log_likelihood = w['l']

c = test_weight(test_data, test_labels, weights)

accuracy = c["correct"]/(c["correct"] + c["wrong"])

print("Negative Log Likelihood")
print(log_likelihood)
print("")
print("Accuracy")
print(accuracy)
