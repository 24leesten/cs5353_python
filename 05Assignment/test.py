'''
Created on Sep 23, 2016

@author: Leland Stenquist
'''
from ID3 import scan, id3, print_tree, test_id3
from ID3_Helper import ensemble_data

r = scan("res/test/train.labels","res/test/train.data",50)
#r = scan("res/madelon/madelon_train.labels","res/madelon/madelon_train.data",25)
e = ensemble_data(r['d'],r['l'])
t = id3(e['d'],e['l'],r['a'],8)
print_tree(t)
test = test_id3(t,r['d'],r['l'],r['a'])
print(test)


