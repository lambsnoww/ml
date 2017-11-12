#_*_coding:utf-8_*_

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

if __name__ == "__main__":
    d1 = pd.read_csv("Youtube01-Psy.csv")
    d2 = pd.read_csv("Youtube02-KatyPerry.csv")
    d3 = pd.read_csv("Youtube03-LMFAO.csv")
    d4 = pd.read_csv("Youtube04-Eminem.csv")
    d4 = pd.read_csv("Youtube05-Shakira.csv")
    lk = tw.hasLink(d1["CONTENT"])
    ls = pd.Series(lk)
    d1 = pd.DataFrame({"CONTENT": d1["CONTENT"], "CLASS": d1["CLASS"], "LINK": ls})

    # word
    word = Word(10, pd.Series.tolist(d1["CONTENT"]))

    # write sentences to file1.txt for Semafor to process
    f = open('file1.txt', 'w')
    for sen in d1["CONTENT"]:
        f.write(sen + '\n')
    f.close()

    # frame
    frame = Frame(10, 'f1out.txt')

    w = pd.DataFrame(word.vector)
    f = pd.DataFrame(frame.vector)
    all = np.hstack((word.vector, frame.vector))
    a = pd.DataFrame(all)
    a = pd.concat([a, d1[["CLASS"]], d1[["LINK"]]], axis=1)
    #a = pd.concat([a, d1[['CONTENT']]], axis=1)
    print len(a)
    print type(a)
    print len(a.xs(0))

    tz_counts = a['CLASS'].value_counts()
    #print tz_counts[:2]
    #print a['LINK'].value_counts()
    #print a.columns()

    haslink = a[a["LINK"] == 1]
    haslink2 = a[a["LINK"] == 1].drop('LINK', axis=1)
    nolink = a[a["LINK"] == 0]
    nolink2 = a[a["LINK"] == 0].drop('LINK', axis=1)

    print haslink['CLASS'].value_counts()

    #np.set_printoptions(threshold=np.inf)
    #print haslink[haslink['CLASS'] == 0]['CONTENT']

    x = nolink2.iloc[:, -1].values.tolist()
    y = nolink2.iloc[:, 0:-1].values.tolist()

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    x_train = StandardScaler(x_train)
    clf = AdaBoostClassifier(n_estimators=100)
    y_pred = clf.fit(x_train, y_train).predict(x_test)















    '''
    print len(word.vector)
    print len(frame.vector)
    all = np.hstack((word.vector, frame.vector))
    a = all.tolist()
    w = pd.Series(word.vector)
    f = pd.Series(frame.vector)
    
    
    
    
    
    d1 = pd.DataFrame({"CONTENT": d1["CONTENT"], "WORD": w, "FRAME": f, "WF": a, "LINK": ls, "CLASS": d1["CLASS"]})
    dw = pd.DataFrame({"WORD": w, "CLASS": d1['CLASS']})
    df = pd.DataFrame({"FRAME": f, "CLASS": d1['CLASS']})
    da = pd.DataFrame({"WF": a, "CLASS": d1['CLASS']})
    '''
    #########
    '''
    x = np.array(np.hstack((frame.vector, word.vector)))
    y = np.array(d1["CLASS"].tolist())
    xy = np.hstack(x, y, lk)
    da = pd.DataFrame({"ATT": x, "CLASS": y})
    # y.reshape(1, -1)
    #print len(x)
    #print len(x[0])
    #print len(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    x_train = StandardScaler(x_train)
    clf = AdaBoostClassifier(n_estimators=100)
    y_pred = clf.fit(x_train, y_train).predict(x_test)
    print y_pred
    print "MSE: ", metrics.mean_squared_error(y_test, y_pred)
    print "RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    ev.outcome(y_pred, y_test)
    '''
    ########
    '''
    data = np.hstack((all, np.array(d1["CLASS"].tolist()).shape(-1, 1), np.array(d1["LINK"].tolist()).shape(-1, 1)))
    data1 = np.array([x for x in data if x[-1] == 0])
    data1 = np.delete(data1, -1, axis=1)
    data2 = np.array([x for x in data if x[-1] == 1])
    data2 = np.delete(data1, -1, axis=1)

    count_link = len(data2)
    print count_link
    print sum(data[-1])
    print len(data[-1])
    '''


    '''
    X = d1['WORD']
    y = d1[['CLASS']]
    print X.head()
    print y.head()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print X_train.shape
    print y_train.shape
    print X_test.shape
    print y_test.shape
    clf = AdaBoostClassifier(n_estimators=100)
    y_pred = clf.fit(X_train, y_train)
    print "MSE:", metrics.mean_squared_error(y_test, y_pred)
    '''


    # frame = Frame(10, pd.Series.tolist(d1["CONTENT"]))
