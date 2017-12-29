#_*_coding:utf-8_*_
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# the stacked classifier
from stacked.stacked_generalization.lib.stacking import StackedClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tools
from random import randint

if __name__ == '__main__':

    df = pd.read_csv('sem_features.txt', header=None)

    f = open('labels.txt', 'r')
    labels = f.read().split()
    # do not forget to transfer chr to int
    labels = [ord(i)-ord('0') for i in labels]
    f.close()

    f = open('sens_xlink.txt', 'r')
    sens = f.readlines()
    f.close()





    x = df.values
    y = labels

    seed = randint(1, 10000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_train_sens, x_test_sens, y_train, y_test = train_test_split(sens, y, test_size=0.2, random_state=seed)

    # MLP(10)和SVM(001,100)
    #clf = MLPClassifier(hidden_layer_sizes=10)
    clf = svm.SVC(gamma=0.001, C=100)
    #clf = LinearDiscriminantAnalysis(n_components=1)
    #clf = GaussianNB()
    #clf = BernoulliNB()
    #clf = tree.DecisionTreeClassifier()
    #clf = AdaBoostClassifier(n_estimators=100)
    #clf = KNeighborsClassifier()
    # clf = AdaBoostClassifier(n_estimators=100)

    #from stacked.stacked_generalization.lib.stacking import StackedClassifier
    #bclf = KNeighborsClassifier()
    #clfs = [GaussianNB(), BernoulliNB(), tree.DecisionTreeClassifier()]
    #clf = StackedClassifier(bclf, clfs)

    clf.fit(x_train, y_train)

    y_pred_train = clf.predict(x_train)
    print "On train data:"
    tools.evaluate(y_pred_train, y_train)

    print "On test data:"
    y_pred = clf.predict(x_test)
    tools.evaluate(y_pred, y_test)

    for i in range(len(y_test)):
        if y_test[i] == 0 and y_pred[i] == 1:
            print '1, 0'
            print x_test_sens[i]
    for i in range(len(y_test)):
        if y_test[i] == 1 and y_pred[i] == 0:
            print '0, 1'
            print x_test_sens[i]






