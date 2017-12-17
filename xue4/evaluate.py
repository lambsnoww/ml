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
        elif (a[i] == 1 and p[i] != 1):
            FN += 1
        elif (a[i] != 1 and p[i] == 1):
            FP += 1
        elif (a[i] != 1 and p[i] != 1):
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
    '''
    print "****************************************"
    print "from/to  spam  no-spam"
    print "spam      %4d  %4d"%(TP, FN)
    print "no-spam   %4d  %4d"%(FP, TN)
    print "****************************************"
    f = open("OUTCOMES.txt", 'a')
    f.write("accuracy, precision, recall, F-value:\n")
    f.write('(%.4f, %.4f, %.4f, %.4f)\n'%(A, P, R, F))
    f.write("****************************************\n")
    f.write("from/to\tspam\tno-spam\n")
    f.write("spam\t%4d\t%4d\n"%(TP, FN))
    f.write("no-spam\t%4d\t%4d\n"%(FP, TN))
    f.write("****************************************\n")
    f.close()
    '''
    return A, P, R, F


def outcome2(p, a, p2, a2):
    n = len(a)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(0, n):
        if (a[i] == 1 and p[i] == 1):
            TP += 1
        elif (a[i] == 1 and p[i] != 1):
            FN += 1
        elif (a[i] != 1 and p[i] == 1):
            FP += 1
        elif (a[i] != 1 and p[i] != 1):
            TN += 1
    for i in range(0, len(a2)):
        if (a2[i] == 1 and p2[i] == 1):
            TP += 1
        elif (a2[i] == 1 and p2[i] != 1):
            FN += 1
        elif (a2[i] != 1 and p2[i] == 1):
            FP += 1
        elif (a2[i] != 1 and p2[i] != 1):
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
    '''
    print "****************************************"
    print "from/to\tspam\tno-spam"
    print "spam\t%4d\t%4d"%(TP, FN)
    print "no-spam\t%4d\t%4d"%(FP, TN)
    print "****************************************"

    f = open("OUTCOMES.txt", 'a')
    f.write("accuracy, precision, recall, F-value:\n")
    f.write('(%.4f, %.4f, %.4f, %.4f)\n'%(A, P, R, F))
    f.write("****************************************\n")
    f.write("from/to\tspam\tno-spam\n")
    f.write("spam\t%4d\t%4d\n"%(TP, FN))
    f.write("no-spam\t%4d\t%4d\n"%(FP, TN))
    f.write("****************************************\n")
    f.close()
    '''

    return A, P, R, F


