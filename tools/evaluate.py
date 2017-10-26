#_*_coding:utf-8_*_

def outcome(p, a):
    n = len(a)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(0, n):
        if (a[i] == 1 and p[i] == 1):
            TP += 1
        elif (a[i] == 1 and p[i] == 0):
            FN += 1
        elif (a[i] == 0 and p[i] == 1):
            FP += 1
        elif (a[i] == 0 and p[i] == 0):
            TN += 1

    #accuracy
    A = float(TP + TN) / (TP + FN + FP + TN)
    #precision
    P = float(TP) / (TP + FP)
    #recall
    R = float(TP) / (TP + FN)
    #F-value
    F = 2 * P * R / (P + R)
    print "accuracy, precision, recall, F-value:"
    print (A, P, R, F)
    return A, P, R, F


