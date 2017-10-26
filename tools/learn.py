#_*_coding:utf-8_*_

import numpy as np
import csv
from sklearn import svm
import tools.evaluate as ra
from sklearn.ensemble import RandomForestClassifier
import string
import random
import tools

def loadDataRandom(per):
    #per - percentage
    data = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/attr.txt"), delimiter=",", skiprows=0)
    label = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/labels.txt"), delimiter=",", skiprows=0)
    n = len(data)
    m1 = int(n * per)
    m2 = n - m1
    trainIndex = random.sample(range(0, n), m1)
    trainIndex = np.sort(trainIndex, axis=0)
    trainData = np.array([data[i] for i in trainIndex])
    trainLabel = np.array([label[i] for i in trainIndex])
    testIndex = np.setxor1d([i for i in range(n)], trainIndex)
    testData = np.array([data[i] for i in testIndex])
    testLabel = np.array([label[i] for i in testIndex])
    print "****************"
    print testData
    print "&&&&&&&&&&&&&&&&"
    return trainData, trainLabel, testData, testLabel
def loadDataSequential(per):
    data = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/attr.txt"), delimiter=",", skiprows=0)
    label = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/labels.txt"), delimiter=",", skiprows=0)
    n = len(data)
    m1 = int(n * per)
    m2 = n - m1
    trainData = data[0:m1]
    trainLabel = label[0:m1]
    testData = data[m1:]
    testLabel = label[m1:]
    return trainData, trainLabel, testData, testLabel



if __name__ == "__main__":
    #trainData, trainLabel, testData, testLabel = loadDataRandom(0.7)
    trainData, trainLabel, testData, testLabel = loadDataSequential(0.5)
    clf = svm.SVC()
    clf.fit(trainData, trainLabel)
    pre = clf.predict(testData)
    out = abs(testLabel - pre)
    #print 1 - float(sum(out))/(len(testData))
    print out
    print sum(out)
    print len(testData)
    f = open("/Users/linxue/PycharmProjects/ml/resources/data.txt")
    a = f.readlines()
    n = len(out)


    #evaluate the outcome of the classifier
    tools.evaluate.outcome(pre, testLabel)




