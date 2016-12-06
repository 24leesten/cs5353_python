import numpy as np
import random


def rand():
    return random.random()


def gradient_descent_logistic_reg(training_data, y_vals, epochs=1, sigma=4, bias=1, r=0.01):
    rand_weights = False
    range_td = list(range(len(y_vals)))

    # Let's get all the weights
    row_len = len(training_data[0])
    weights = np.zeros(len(training_data[0])).tolist()

    for i in range(row_len):
        weights[i] = rand()

    # Let's add in the bias
    if rand_weights:
        bias = rand()
    weights.insert(0, bias)

    # insert the y value as the bias for these variables
    i = 0
    for x in training_data:
        x.insert(0, y_vals[i])
        i += 1

    objectives = []

    for epoch in range(epochs):

        likelihood = 0

        random.shuffle(range_td)

        gradient = 0
        w = np.array(weights)

        for row in range_td:
            x = np.array(training_data[row])
            y = y_vals[row]
            # These are version of the gradient I went through. They were giving me overflow warnings and errors
            #    numerator = (-y * x * np.exp(-y * np.dot(w, x)))
            #    denominator = (1 + np.exp(-y * np.dot(w, x)))
            #    gradient = (numerator / denominator) + (2 * w) / (delta ** 2)
            #    gradient =  (-y * x)*(1-(1/(1 + np.exp(-y * np.dot(w, x))))) + (2 * w) / (delta ** 2)
            # Simplified version of the equation in the algorithm

            var1 = -y * x/(np.exp(y * np.dot(w, x)) + 1)
            var2 = (2 * w) / (sigma ** 2)

            gradient = var1 + var2
            w = w - (r * gradient)
            likelihood += -np.log(1+np.exp(-y*np.dot(w,x)))

        weights = w.tolist()

        update = (1/(sigma^2)) * np.dot(w,w)
        objective = update + likelihood
        objectives.append(objective)

    return {'w':weights,'o':objectives}


def test_weight(training_data, y_vals, w):
    # insert the y value as the bias for these variables
    i = 0
    for x in training_data:
        x.insert(0, y_vals[i])
        i += 1

    # make sure that the vectors are the same dimension.
    if (len(w) < len(training_data[0])):
        while (len(w) < len(training_data[0])):
            w.append(0)
    if (len(w) > len(training_data[0])):
        while (len(w) > len(training_data[0])):
            w.pop()

    range_td = list(range(len(y_vals)))

    correct = 0
    wrong = 0

    for row in range_td:
        y = y_vals[row]
        w = np.array(w)
        x = np.array(training_data[row])
        if y == (np.sign(np.dot(w, x))):
            correct += 1
        else:
            wrong += 1

    return {"correct": correct, "wrong": wrong}
