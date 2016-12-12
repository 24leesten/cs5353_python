import warnings
warnings.simplefilter("ignore", RuntimeWarning)
from Helper import cross_validation_scan, cross_validation

files = cross_validation_scan()
sigmas = [0.1, 0.25, 0.5, 0.75, 1 , 1.5, 2, 4, 8, 16, 32, 64]
epochs = [1, 5, 10, 20, 50]

for sigma in sigmas:
    accs = []
    print("==== SIGMA: " + str(sigma) + " ====")
    print("  == EPOCHS ==")
    for epoch in epochs:
        acc = cross_validation(files, epoch, sigma)
        accs.append(acc)

    for i in range(len(accs)):
        print("      - " + str(epochs[i]) + ": \t" + str(accs[i]))
