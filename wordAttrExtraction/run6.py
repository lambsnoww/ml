#_*_coding:utf-8_*_
#使用了word和frame的前10高频及后10调频特征提取，
# 并使用stacking和adaboost算法进行训练，92%左右

import pandas as pd
import numpy as np
import tools.wordProcess as tw
from Word import *
from Frame import *
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
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import string
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import copy
from stacked.stacked_generalization.lib.stacking import StackedClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def getFrequency(dictlist1, dictlist2):
    count1 = copy.deepcopy(dictlist1)
    count2 = copy.deepcopy(dictlist2)

    for l in count1:
        nn = len(l)
        if nn == 0:
            continue
        for t in l.keys():
            l[t] = float(l[t]) / nn
    for l in count2:
        nn = len(l)
        if nn == 0:
            continue
        for t in l.keys():
            l[t] = float(l[t]) / nn * (-1)

    d = defaultdict()
    for l in count1:
        for items in l.items():
            if items[0] in d:
                d[items[0]] += items[1]
            else:
                d[items[0]] = items[1]
    for l in count2:
        for items in l.items():
            if items[0] in d:
                d[items[0]] += items[1]
            else:
                d[items[0]] = items[1]

    #print d
    return d


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
    d = pd.read_csv("Youtube.csv")

    lk = tw.hasLink(d["CONTENT"])
    ls = pd.Series(lk)
    d = pd.DataFrame({"CONTENT": d["CONTENT"], "CLASS": d["CLASS"], "LINK": ls})

    dp = d[d["CLASS"] == 1]
    dn = d[d["CLASS"] == 0]

    # word
    #word = Word(10, pd.Series.tolist(d["CONTENT"]))
    pword = Word(10, pd.Series.tolist(dp["CONTENT"]))
    nword = Word(10, pd.Series.tolist(dn["CONTENT"]))

    #print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
    #print pword.countlist
    #print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
    #print nword.countlist
    #print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

    wordFreq = getFrequency(pword.countlist, nword.countlist)

    pframe = Frame(100, 'psensFrame.txt')
    nframe = Frame(100, 'nsensFrame.txt')

    frameFreq = getFrequency(pframe.countlist, nframe.countlist)

    wf = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
    ff = sorted(frameFreq.items(), key=lambda x: x[1], reverse=True)
    wn = 10; fn = 10
    fl1 = []; wl1 = []
    fl2 = []; wl2 = []

    for i in range(wn):
        wl1.append(wf[i][0])
    for i in range(fn):
        fl1.append(ff[i][0])
    for i in range(wn):
        wl2.append(wf[-i-1][0])
    for i in range(fn):
        fl2.append(ff[-i-1][0])

    #print wl1
    #print wl2
    #print fl1
    #print fl2

    wl = copy.deepcopy(wl1)
    fl = copy.deepcopy(fl1)
    wl.extend(wl2) # the words list of our consideration
    fl.extend(fl2) # the frames list of our consideration
    #print "*************"
    #print wl
    #word = Word(10, pd.Series.tolist(d["CONTENT"]))
    frame = Frame(100, 'allsensFrame.txt')

    f = open("allsens.txt", 'r')
    a = f.readlines()
    wordlist =[]
    for line in a:
        l = [0] * len(wl)
        for i in range(len(wl)):
            l[i] += line.lower().count(wl[i])
        wordlist.append(l)
    #print wordlist
    f.close()

    # frame.countlist
    framelist = []

    for framedict in frame.countlist:
        l = [0] * len(fl)
        for i in range(len(fl)):
            if fl[i] in framedict:
                l[i] = framedict[fl[i]]
        framelist.append(l)
    #print framelist
    x1 = copy.deepcopy(wordlist)
    x2 = copy.deepcopy(framelist)
    #x = np.array(x1)
    x = np.array(np.hstack((x1, x2)))
    lk = np.array(list(d['LINK']))
    print x.shape
    print lk.shape
    x = np.hstack((x, lk.reshape(-1,1)))
    x1 = np.hstack((x1, lk.reshape(-1,1)))
    y = np.array(list(d["CLASS"]))
    print x.shape
    print y.shape
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=2)
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y, test_size=0.33, random_state=2)


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

    y_pred = clf.fit(x_train, y_train).predict(x_test)
    y1_pred = clf.fit(x1_train, y1_train).predict(x1_test)
    ev.outcome(y_pred, y_test)
    ev.outcome(y1_pred, y1_test)



















    '''
    # before this step, make sure to write sentences to file1.txt for Semafor to process
    
    f = open('allsens.txt', 'w')
    for sen in d["CONTENT"]:
        f.write(sen + '\n')
    f.close()
    #positive
    fp = open('psens.txt', 'w')
    for sen in dp["CONTENT"]:
        fp.write(sen + '\n')
    fp.close()
    #negative
    fn = open('nsens.txt', 'w')
    for sen in dn["CONTENT"]:
        fn.write(sen + '\n')
    fn.close()
    
    # frame
    #frame = Frame(100, 'allsensFrame.txt')
    #frame = Frame(10, 'f1out.txt')
    pframe = Frame(100, 'psensFrame.txt')
    nframe = Frame(100, 'nsensFrame.txt')
    
    print "VECTORS"
    print "WORD.VECTOR"
    print word.vector
    print "FRAME.VECTOR"
    print frame.vector
    
    #w = pd.DataFrame(word.vector)
    #f = pd.DataFrame(frame.vector)
    ##############
    if (frameOrNot == 1):
        all = np.hstack((word.vector, frame.vector))
    elif (frameOrNot == 0):
        all = word.vector
    elif (frameOrNot == -1):
        all = frame.vector
    a = pd.DataFrame(all)
    a = pd.concat([a, d[["CLASS"]], d[["LINK"]]], axis=1)

    tz_counts = a['CLASS'].value_counts()
    # print tz_counts[:2]
    # print a['LINK'].value_counts()
    # print a.columns()

    haslink = a[a["LINK"] == 1]
    a2 = list(haslink["CLASS"])
    p2 = list(haslink["LINK"])
    haslink2 = a[a["LINK"] == 1].drop('LINK', axis=1)
    nolink = a[a["LINK"] == 0]
    nolink2 = a[a["LINK"] == 0].drop('LINK', axis=1)

    print haslink['CLASS'].value_counts()


    x = nolink2.iloc[:, 0:-1].values
    y = nolink2.iloc[:, -1].values
    
    pca = PCA(n_components=10)
    newData = pca.fit_transform(x)
    print "SHAPE@@@@@@@@@@@@@@@@@@@@@@@@@"
    print x.shape
    print newData.shape
    x = newData
    

    #y = list(nolink2['CLASS'])
    #y = nolink2.iloc[:, -1].values

    print len(x)
    print len(x[0])
    print len(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
    # x_train = StandardScaler(x_train)


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
    elif method == "KNN":
        clf = KNeighborsClassifier()
utcome2(y_pred, y_test, p2, a2)
    print 'frame: ' + str(frameOrNot) + ', method: ' + method
    y_pred = clf.fit(x_train, y_train).predict(x_test)
    #ev.outcome(y_pred, y_test)
    f = open('OUTCOMES.txt', 'a')
    f.write('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')
    f.write('frame: ' + str(frameOrNot) + ', method: ' + method + '\n')
    f.close()

    ev.outcome2(y_pred, y_test, p2, a2)

    print (metrics.classification_report(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))
    #print y_pred
    #print y_test



    '''