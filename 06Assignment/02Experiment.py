import warnings
warnings.simplefilter("ignore", RuntimeWarning)
from Helper import cross_validation_scan, cross_validation

files = cross_validation_scan()
sigmas = [0.5 , 1 , 1.5, 2, 4, 8, 16, 32]
epochs = [1, 5, 10]

for sigma in sigmas:
    print("====SIGMA: " + str(sigma) + " ====")
    for epoch in epochs:
        print("  ==EPOCH: " + str(epoch) + " ==")
        acc = cross_validation(files, epoch, sigma)
        print("    Accuracy: " + str(acc))
