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