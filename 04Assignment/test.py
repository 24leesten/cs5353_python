from ID3 import scan, id3, print_tree, test_id3

r = scan("res/test/test.labels","res/test/test.data")
r = scan("res/madelon/madelon_train.labels","res/madelon/madelon_train.data",50)
t = id3(r['d'],r['l'],r['a'])
print("here")
#print_tree(t)
#test = test_id3(t,r['d'],r['l'],r['a'])
#print(test)


