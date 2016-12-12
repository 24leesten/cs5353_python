'''
Created on Sep 23, 2016

@author: Leland Stenquist
'''

import random

def ensemble_data(data_set,labels):
    size = len(labels)

    new_data_set = []
    new_labels = []

    for i in range(size):
        rand = random.randrange(size)
        new_data_set.append(data_set[rand])
        new_labels.append(labels[rand])

    return {'d':new_data_set,'l':new_labels}

def get_data(trees, data_set):

    new_data_set = []
    for row in data_set:
        data_row = {}
        tree_count = 0
        for tree in trees:
            node = tree
            while not node.leaf:
                val = row[node.attribute]
                for child in node.children:
                    if child.link == val:
                        node = child
                        break
                if(len(node.children)) > 0:
                    node = node.children[0]
                else:
                    break

            label = 1
            if node.leaf:
                label = node.label

            data_row[tree_count] = label
            tree_count += 1
        new_data_set.append(data_row)

    return new_data_set