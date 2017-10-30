#_*_coding:utf-8_*_

import numpy as np
import csv
import scipy as sp

from sklearn import preprocessing
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier


import tools.evaluate as ra
from sklearn.ensemble import RandomForestClassifier
import string
import random
import tools


def loadDataRandom(per, type):
    #per - percentage
    if (type == "frame"):
        data = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/attr_all.txt"), delimiter=",", skiprows=0)
    else:
        data = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/attr.txt"), delimiter=",", skiprows=0)
    label = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/labels.txt"), delimiter=",", skiprows=0)
    print data
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
def loadDataSequential(per, type):
    if type == "frame":
        data = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/attr_all.txt"), delimiter=",", skiprows=0)
    else:
        data = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/attr.txt"), delimiter=",", skiprows=0)
    label = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/labels.txt"), delimiter=",", skiprows=0)
    print data
    n = len(data)
    m1 = int(n * per)
    m2 = n - m1
    trainData = data[0:m1]
    trainLabel = label[0:m1]
    testData = data[m1:]
    testLabel = label[m1:]
    return trainData, trainLabel, testData, testLabel

def printOutcomes(p, pre, testLabel, frameOrNot, method, trainOrTest, loadType, scaled):
    f = open("/Users/linxue/PycharmProjects/ml/resources/OUTCOMES.txt", "a")
    if trainOrTest == "train":
        f.write("\n**************************************************************************************************\n")
    else:
        f.write("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
    sc = "non-scaled"
    if scaled:
        sc = "scaled"
    s = trainOrTest + ',' + str(p) + ',' + frameOrNot + ',' + method + ',' + loadType + ',' + sc + '\n'
    f.write(s)
    f.write(str(ra.outcome(pre, testLabel)))
    if trainOrTest == "test":
        f.write("\n**************************************************************************************************\n")
    f.close()


if __name__ == "__main__":
    ############################################################
    # paramaters
    p = 0.9
    scaled = False
    frameOrNot = "non-frame" # use FrameNet or not
    loadType = "Sequential"
    #method = "Decision tree"
    #method = "SVM"
    #method = "GaussianNB"
    #method = "BernoulliNB"
    method = "AdaBoost"
    ###########################################################

    if loadType == "Random":
        trainData, trainLabel, testData, testLabel = loadDataRandom(p, frameOrNot)
    else:
        trainData, trainLabel, testData, testLabel = loadDataSequential(p, frameOrNot)
    if scaled:
        trainData = preprocessing.scale(trainData)
        testData = preprocessing.scale(testData)
    # clf = svm.SVC(gamma=0.001, C=100)

    if method == "SVM":
        clf = svm.SVC()
    elif method == "GaussianNB":
        clf = GaussianNB()
    elif method == "BernoulliNB":
        clf = BernoulliNB()
    elif method == "Decision Tree":
        clf = tree.DecisionTreeClassifier()
    elif method == "AdaBoost":
        clf = AdaBoostClassifier(n_estimators=100)

    pre = clf.fit(trainData, trainLabel).predict(testData)
    trainPre = clf.fit(trainData, trainLabel).predict(trainData)

    printOutcomes(p, trainPre, trainLabel, frameOrNot, method, "train", loadType, scaled)
    printOutcomes(p, pre, testLabel, frameOrNot, method, "test", loadType, scaled)


