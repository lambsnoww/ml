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
# the stacked classifier
from stacked.stacked_generalization.lib.stacking import StackedClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn import datasets, metrics


import tools.evaluate as ra
from sklearn.ensemble import RandomForestClassifier
import string
import random
import tools


def loadDataRandom(per):
    #per - percentage
    dataf = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/attr_all.txt"), delimiter=",", skiprows=0)
    datanf = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/attr.txt"), delimiter=",", skiprows=0)
    label = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/labels.txt"), delimiter=",", skiprows=0)
    print dataf
    n = len(dataf)
    m1 = int(n * per)
    m2 = n - m1
    # generate index once for both frame and non-frame
    trainIndex = random.sample(range(0, n), m1)
    trainIndex = np.sort(trainIndex, axis=0)

    trainData = np.array([dataf[i] for i in trainIndex])
    trainDatan = np.array([datanf[i] for i in trainIndex])
    trainLabel = np.array([label[i] for i in trainIndex])

    testIndex = np.setxor1d([i for i in range(n)], trainIndex)
    testData = np.array([dataf[i] for i in testIndex])
    testDatan = np.array([datanf[i] for i in testIndex])
    testLabel = np.array([label[i] for i in testIndex])
    return trainData, trainDatan, trainLabel, testData, testDatan, testLabel
def loadDataSequential(per, type):
    dataf = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/attr_all.txt"), delimiter=",", skiprows=0)
    datan = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/attr.txt"), delimiter=",", skiprows=0)
    label = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/labels.txt"), delimiter=",", skiprows=0)
    n = len(dataf)
    m1 = int(n * per)
    m2 = n - m1
    trainData = dataf[0:m1]
    trainDatan = datan[0:m1]
    trainLabel = label[0:m1]
    testData = dataf[m1:]
    testDatan = datan[m1:]
    testLabel = label[m1:]
    return trainData, trainDatan, trainLabel, testData, testDatan, testLabel

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
    p = 0.8
    scaled = False
    loadType = "Random"
    #loadType = "Sequential"
    #method = "Decision tree"
    method = "SVM"
    #method = "GaussianNB"
    #method = "BernoulliNB"
    #method = "AdaBoost"
    #method = "Stacking"
    #method Stacking is not usable now
    ###########################################################

    if loadType == "Random":
        trainData, trainDatan, trainLabel, testData, testDatan, testLabel = loadDataRandom(p)
    else:
        trainData, trainDatan, trainLabel, testData, testDatan, testLabel = loadDataSequential(p)
    if scaled:
        trainData = preprocessing.scale(trainData)
        trainDatan = preprocessing.scale(trainDatan)
        testData = preprocessing.scale(testData)
        testDatan = preprocessing.scale(testDatan)
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

    elif method == "Stacking":
        bclf = LogisticRegression(random_state=1)
        clfs = [RandomForestClassifier(n_estimators=40, criterion='gini', random_state=1),
                GradientBoostingClassifier(n_estimators=25, random_state=1),
                RidgeClassifier(random_state=1)]
        clf = StackedClassifier(bclf, clfs)

    print testData
    print "*************************************"
    pre = clf.fit(trainData, trainLabel).predict(testData)
    trainPre = clf.fit(trainData, trainLabel).predict(trainData)

    printOutcomes(p, trainPre, trainLabel, "frame", method, "train", loadType, scaled)
    printOutcomes(p, pre, testLabel, "frame", method, "test", loadType, scaled)
    #score = metrics.accuracy_score(trainLabel, pre)
    #print ("Accuracy: %f" % score)

    pre = clf.fit(trainDatan, trainLabel).predict(testDatan)
    trainPre = clf.fit(trainDatan, trainLabel).predict(trainDatan)

    printOutcomes(p, trainPre, trainLabel, "non-frame", method, "train", loadType, scaled)
    printOutcomes(p, pre, testLabel, "non-frame", method, "test", loadType, scaled)
    #score = metrics.accuracy_score(trainLabel, pre)
    #print ("Accuracy: %f" % score)


