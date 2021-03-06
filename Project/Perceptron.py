import numpy as np
import random


def rand():
    return random.random()


def perceptron(training_data, y_vals, epochs=1, mu=0, bias=1, r=0.01):

    range_td = list(range(len(y_vals)))

    # Let's get all the weights
    row_len = len(training_data[0])
    weights = np.zeros(len(training_data[0])).tolist()
    for i in range(row_len):
        weights[i] = rand()

    # insert the y value as the bias for these variables
    weights.insert(0, bias)
    i = 0
    for x in training_data:
        x.insert(0, y_vals[i])
        i += 1

    for epoch in range(epochs):

        random.shuffle(range_td)

        w = np.array(weights)

        for row in range_td:
            x = np.array(training_data[row])
            y = y_vals[row]

            if y * np.dot(w,x) <= mu :
                gradient = y * x
                w = w + (r * gradient)

        weights = w.tolist()

    return weights


def test_perceptron(training_data, y_vals, w):
    # insert the y value as the bias for these variables
    i = 0
    for x in training_data:
        x.insert(0, 0)
        i += 1

    # make sure that the vectors are the same dimension.
    if len(w) < len(training_data[0]):
        while len(w) < len(training_data[0]):
            w.append(0)
    if len(w) > len(training_data[0]):
        while len(w) > len(training_data[0]):
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
