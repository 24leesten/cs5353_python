import warnings
warnings.simplefilter("ignore", RuntimeWarning)
from Helper import cross_validation_scan, cross_validation_per, cross_validation_log, cross_validation_svm, cross_validation_win

epochs = [1, 5, 10]

# print("===========================")
# print("PERCEPTRON")
# print("===========================")
#
files = cross_validation_scan()
# mus = [0.1, 0.25, 0.5, 0.75, 1 , 1.5]
#
# for mu in mus:
#     accs = []
#     print("==== MU: " + str(mu) + " ====")
#     print("  == EPOCHS ==")
#     for epoch in epochs:
#         acc = cross_validation_per(files, epoch, mu)
#         accs.append(acc)
#
#     for i in range(len(accs)):
#         print("      - " + str(epochs[i]) + ": \t" + str(accs[i]))
#
#
#
# print("===========================")
# print("SVM")
# print("===========================")
#
# Cs = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, 8, 16, 32, 64]
#
# for c in Cs:
#     accs = []
#     print("==== C: " + str(c) + " ====")
#     print("  == EPOCHS ==")
#     for epoch in epochs:
#         acc = cross_validation_svm(files, epoch, c)
#         accs.append(acc)
#
#     for i in range(len(accs)):
#         print("      - " + str(epochs[i]) + ": \t" + str(accs[i]))



# print("===========================")
# print("LOG REG")
# print("===========================")
#
# epochs = [10, 32, 64]
# sigmas = [0.1, 0.5, 1 , 2, 4, 8, 16, 32, 64]
#
# for sigma in sigmas:
#     accs = []
#     print("==== SIGMA: " + str(sigma) + " ====")
#     print("  == EPOCHS ==")
#     for epoch in epochs:
#         acc = cross_validation_log(files, epoch, sigma)
#         accs.append(acc)
#
#     for i in range(len(accs)):
#         print("      - " + str(epochs[i]) + ": \t" + str(accs[i]))

# print("===========================")
# print("WINNOW")
# print("===========================")
#
# epochs = [1, 5, 15, 30]
# R = [1.5, 2, 4, 8]
#
# for r in R:
#     accs = []
#     print("==== SIGMA: " + str(r) + " ====")
#     print("  == EPOCHS ==")
#     for epoch in epochs:
#         acc = cross_validation_win(files, epoch, r)
#         accs.append(acc)
#
#     for i in range(len(accs)):
#         print("      - " + str(epochs[i]) + ": \t" + str(accs[i]))

print("===========================")
print("ID3")
print("===========================")

epochs = [10, 32, 64]
sigmas = [0.1, 0.5, 1 , 2, 4, 8, 16, 32, 64]

for sigma in sigmas:
    accs = []
    print("==== SIGMA: " + str(sigma) + " ====")
    print("  == EPOCHS ==")
    for epoch in epochs:
        acc = cross_validation_log(files, epoch, sigma)
        accs.append(acc)

    for i in range(len(accs)):
        print("      - " + str(epochs[i]) + ": \t" + str(accs[i]))