#_*_coding:utf-8_*_

import pandas as pd
import numpy as np
import tools.wordProcess as tw
from Word import *
from Frame import *
from Frame2 import *
import tools.evaluate as ev

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
from sklearn.preprocessing import StandardScaler
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
from sklearn.model_selection import train_test_split
from sklearn import metrics
import string
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import copy
from stacked.stacked_generalization.lib.stacking import StackedClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def getTf(lst):

    dictlist = []
    alldict = defaultdict(float)
    for sen in lst:
        dict = defaultdict(float)
        words = sen.split()
        for w in words:
            dict[w] += 1
            alldict[w] += 1
        for w in words:
            dict[w] /= len(words)
        dictlist.append(dict)
    keylist = []
    for key in alldict.keys():
        keylist.append(key)

    #print keylist

    n = len(keylist)
    arr = []
    for d in dictlist:
        a = [0] * n
        for ikey in d.keys():
            a[keylist.index(ikey)] = d[ikey]
        arr.append(a)

    word_frame = pd.DataFrame(arr, columns=keylist)
    #print word_frame.head(10)

    return word_frame

def getIdf(lst):
    pass

def trimSens(lst):
    rt = []
    for sen in lst:
        newsen = ''
        words = sen.split()
        for word in words:
            while (word != '' and (word[0] < 'A' or word[0] >'z')):
                word = word[1:]
            while (word != '' and (word[-1] < 'A' or word[-1] >'z')):
                word = word[:-1]
            if word != '':
                word = word.lower()
                newsen = newsen + word + ' '

        rt.append(newsen.strip())

    return rt

def get_features(flst, sens):
    rt = []
    n = len(flst)
    farr = flst.tolist()
    for sen in sens:
        a = [0] * n
        words = sen.split()
        for i in words:
            if i in farr:
                a[farr.index(i)] += 1
        rt.append(a)
    return np.array(rt)


if __name__ == "__main__":

    # 1 - frame and word
    # 0 - word only
    # -1 - frame only
    frameOrNot = 1

    #method = "SVM"
    #method = "GaussianNB"
    #method = "BernoulliNB"
    #method = "Decision Tree"
    method = "AdaBoost"
    #method = "KNN"

    #d = pd.read_csv("Youtube01-Psy.csv")
    #d = pd.read_csv("Youtube02-KatyPerry.csv")
    #d = pd.read_csv("Youtube03-LMFAO.csv")
    #d = pd.read_csv("Youtube04-Eminem.csv")
    #d = pd.read_csv("Youtube05-Shakira.csv")
    d0 = pd.read_csv("Youtube.csv")
    #f = open('allsensFrame.txt', 'r')
    #f0 = pd.Series(f.readlines())
    #f.close()

    lk = tw.hasLink(d0["CONTENT"])
    ls = pd.Series(lk)
    frame = Frame2(10, 'allsensFrame.txt')
    # d0 -- all of the data
    content = trimSens(list(d0['CONTENT']))
    d0 = pd.DataFrame({"CONTENT": content, "FRAME": frame.framelist, "CLASS": d0["CLASS"], "LINK": ls})
    # d1 -- the data without links
    d1 = d0[d0['LINK'] == 0]
    d2 = d0[d0['LINK'] == 1]
    yp = np.array(d2['LINK'])
    yo = np.array(d2['CLASS'])
    #print d1.head()

    x_train, x_test, y_train, y_test = train_test_split(d1[['CONTENT', 'FRAME', 'CLASS']], d1['CLASS'], test_size=0.33, random_state=134432)

    #print len(x_train)
    #print len(y_train)
    #print len(x_test)
    #print len(y_test)

    p = x_train[x_train['CLASS'] == 1]
    n = x_train[x_train['CLASS'] == 0]

    p_wtf = getTf(list(p['CONTENT']))
    n_wtf = getTf(list(n['CONTENT']))
    n_wtf = n_wtf * -1

    tf = pd.merge(p_wtf, n_wtf, how='outer')
    tf = tf.fillna(0)

    tf.loc['Row_sum'] = tf.apply(lambda x: x.sum())
    tf = abs(tf)
    tf = tf.sort_values(by='Row_sum', axis=1, ascending=False)

    word_features = tf.columns.values[0:100]
    ###################

    p = x_train[x_train['CLASS'] == 1]
    n = x_train[x_train['CLASS'] == 0]

    p_wtf = getTf(list(p['FRAME']))
    n_wtf = getTf(list(n['FRAME']))
    n_wtf = n_wtf * -1

    tf = pd.merge(p_wtf, n_wtf, how='outer')
    tf = tf.fillna(0)

    tf.loc['Row_sum'] = tf.apply(lambda x: x.sum())
    tf = abs(tf)
    tf = tf.sort_values(by='Row_sum', axis=1, ascending=False)

    frame_features = tf.columns.values[0:100]
    ########## features found finished! #############




    xtrain1 = get_features(word_features, x_train['CONTENT'])
    xtrain2 = get_features(frame_features, x_train['FRAME'])
    xtrain = np.concatenate((xtrain1, xtrain2), axis=1)
    #xtrain = xtrain1

    xtest1 = get_features(word_features, x_test['CONTENT'])
    xtest2 = get_features(frame_features, x_test['FRAME'])
    xtest = np.concatenate((xtest1, xtest2), axis=1)
    #xtest = xtest1

    ytrain = np.array(y_train.tolist())
    ytest = np.array(y_test.tolist())
    print len(xtrain)
    print len(ytrain)
    print ytrain
    print y_train
    print type(y_train)
    print y_train.index.values
    #print xtrain


    #clf = svm.SVC()
    #clf = GaussianNB()
    #clf = BernoulliNB()
    #clf = tree.DecisionTreeClassifier()
    #clf = AdaBoostClassifier(n_estimators=100)
    #clf = KNeighborsClassifier()
    #clf = AdaBoostClassifier(n_estimators=100)

    #from stacked.stacked_generalization.lib.stacking import StackedClassifier
    bclf = KNeighborsClassifier()
    clfs = [GaussianNB(), BernoulliNB(), tree.DecisionTreeClassifier()]
    clf = StackedClassifier(bclf, clfs)
    print x_train
    print y_train

    classifier = clf.fit(xtrain, ytrain)
    ypred = classifier.predict(xtest)
    # ypred = clf.fit(xtrain, ytrain).predict(xtest)
    ev.outcome(ypred, ytest)
    ev.outcome2(ypred, ytest, yp, yo)


    print "___________________________________________________________"
    classifier = clf.fit(xtrain1, ytrain)
    ypred1 = classifier.predict(xtest1)
    # ypred = clf.fit(xtrain, ytrain).predict(xtest)
    ev.outcome(ypred1, ytest)
    ev.outcome2(ypred1, ytest, yp, yo)


    print (metrics.classification_report(ytest, ypred))
    print (metrics.confusion_matrix(ytest, ypred))
    print (metrics.classification_report(ytest, ypred1))
    print (metrics.confusion_matrix(ytest, ypred1))










