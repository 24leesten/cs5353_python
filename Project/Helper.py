from LogRegClass import gradient_descent_logistic_reg, test_log_reg_class
from Scanner import scan


def cross_validation_scan():
    # v1_train = scan("res/x_validation/v1.train")
    # v1_test = scan("res/x_validation/v1.test")
    # v2_train = scan("res/x_validation/v2.train")
    # v2_test = scan("res/x_validation/v2.test")
    # v3_train = scan("res/x_validation/v3.train")
    # v3_test = scan("res/x_validation/v3.test")
    # v4_train = scan("res/x_validation/v4.train")
    # v4_test = scan("res/x_validation/v4.test")
    # v5_train = scan("res/x_validation/v5.train")
    # v5_test = scan("res/x_validation/v5.test")
    # v6_train = scan("res/x_validation/v6.train")
    # v6_test = scan("res/x_validation/v6.test")

    v1_train = scan("res/AHU27.csv")
    v1_test = scan("res/AHU27.csv")
    v2_train = scan("res/AHU27.csv")
    v2_test = scan("res/AHU27.csv")
    v3_train = scan("res/AHU27.csv")
    v3_test = scan("res/AHU27.csv")
    v4_train = scan("res/AHU27.csv")
    v4_test = scan("res/AHU27.csv")
    v5_train = scan("res/AHU27.csv")
    v5_test = scan("res/AHU27.csv")
    v6_train = scan("res/AHU27.csv")
    v6_test = scan("res/AHU27.csv")

    return {"v1": [v1_train, v1_test], "v2": [v2_train, v2_test], "v3": [v3_train, v3_test],
            "v4": [v4_train, v4_test], "v5": [v5_train, v5_test], "v6": [v6_train, v6_test]}


def cross_validation(files, epochs, delta, bias=0, r=0.01):
    accuracy = []

    for file in files:
        train_file = files[file][0]
        test_file = files[file][1]

        training_set = train_file['d']
        training_lbl = train_file['l']
        test_set = test_file['d']
        test_lbl = test_file['l']

        w = gradient_descent_logistic_reg(training_set, training_lbl, epochs, delta, bias, r)
        c = test_log_reg_class(test_set, test_lbl, w)
        accuracy.append(c["correct"] / (c["correct"] + c["wrong"]))

    return sum(accuracy) / len(accuracy)
