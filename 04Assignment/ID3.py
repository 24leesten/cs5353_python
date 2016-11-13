import csv
import ast
import numpy
import math


class Tree(object):
    def __init__(self):
        self.children = []
        self.link = None
        self.attribute = None
        self.label = None
        self.leaf = False


def scan(labels_file, Attributes_file, numeric=None):
    data_set = [];
    labels = [];
    attributes = [];

    with open(labels_file) as csvfile:
        # Read in the CSV
        labels_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

        # fill in y_vals with CSV data
        for row in labels_reader:
            labels.append(int(row.pop(0)))

    with open(Attributes_file) as csvfile:
        # Read in the CSV
        data_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

        if numeric is None:
            # fill in training_data with CSV data
            for row in data_reader:
                count = 0
                dict_str = '{'
                for val in row:
                    attributes.append(count)
                    dict_str = dict_str + str(count) + ":" + val + ','
                    count += 1
                dict_str = dict_str[:-1] + '}'
                data_set.append(ast.literal_eval(dict_str))
        else:
            # fill in training_data with CSV data
            for row in data_reader:
                count = 0
                dict_str = '{'
                for val in row:
                    attributes.append(count)
                    val = str(int(math.floor(float(val) / numeric)))
                    dict_str = dict_str + str(count) + ":" + val + ','
                    count += 1
                dict_str = dict_str[:-1] + '}'
                data_set.append(ast.literal_eval(dict_str))

    return {'d': data_set, 'l': labels, 'a': attributes}


def get_column(data_set, col):
    column = []
    for row in data_set:
        column.append(row[col])
    return column


def get_label(labels):
    return max(labels.count(-1), labels.count(1))
    pass


def id3(data_set, labels, attributes):
    unique_lbl = numpy.unique(labels)
    if len(unique_lbl) == 1:
        t = Tree()
        t.leaf = True
        t.label = unique_lbl[0]
    else:
        t1 = Tree()
        entropy = 0
        probs = []
        info_gain = []
        idx = 0
        # Get the main entropy
        for lbl in unique_lbl:
            probs.append(labels.count(lbl) / len(labels))
            entropy = entropy + (-probs[idx] * math.log2(probs[idx]))
            idx += 1
        # Get the entropy for each attribute
        for attribute in attributes:  # TODO : could be probelmatic
            # get the unique attribute
            column = get_column(data_set, attribute)
            unique_vals = numpy.unique(column)
            expected_entropy = 0
            for val in unique_vals:
                pos = neg = 0
                length = column.count(val)
                entropy = 0
                count = 0
                for c in column:
                    if c == val:
                        if (labels[count] == 1):
                            pos += 1
                        else:
                            neg += 1
                    count += 1
                if (neg != 0 and pos != 0):
                    entropy = (-(pos / length) * math.log2(pos / length)) * (-(neg / length) * math.log2(neg / length))
                expected_entropy = expected_entropy + entropy * (length / len(column))
            info_gain.append(entropy - expected_entropy)

        attribute = numpy.argmax(info_gain)
        t1.attribute = attribute
        attributes.remove(attribute)

        unique_vals = numpy.unique(get_column(data_set, attribute))
        for val in unique_vals:
            count = 0
            new_data_set = []
            new_labels = []
            for row in data_set:
                if row[attribute] == val:
                    new_data_set.append(row)
                    new_labels.append(labels[count])
                count += 1
            if len(new_data_set) == 0:
                leaf = Tree
                leaf.link = val
                leaf.leaf = True
                leaf.label = get_label(labels)
                t1.children.append(leaf)
            else:
                t = id3(new_data_set, new_labels, attributes)
                t.link = val
                t1.children.append(t)
        t = t1
    return t


def print_child(child, count):
    count += 1
    tab = ""
    if (child.leaf):
        for i in range(0, count):
            tab += "--"
        print(tab + "[" + str(child.label) + "]")
        return
    else:
        for i in range(0, count):
            tab += "--"
        print(tab + str(child.link) + "|" + str(child.attribute))
        for c in child.children:
            print_child(c, count)


def print_tree(tree):
    print(tree.attribute)
    children = tree.children
    for child in children:
        print_child(child, 0)


def test_id3(tree, data_set, labels, attributes):
    correct = 0
    wrong = 0

    count = 0
    for row in data_set:
        node = tree
        while not node.leaf:
            val = row[node.attribute]
            for child in node.children:
                if (child.link == val):
                    node = child

        if labels[count] == node.label:
            correct += 1
        else:
            wrong += 1
        count += 1

    return {'c': correct, 'w': wrong, 'a': correct / count, 'e': wrong / count}