#_*_coding:utf-8_*_
# 在5的基础上去掉了停用词，效果提升
# 在6的基础上，随机产生10个样本分类测试，并word和frame各取100时效果好
# 参数：
# 1.frame的个数选择
# 2.word和frame的特征个数选择


from __future__ import division
import pandas as pd
import numpy as np
import tools.wordProcess as tw
from wordAttrExtraction.Word import *
from wordAttrExtraction.Frame import *
from wordAttrExtraction.Frame2 import *
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
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import feature_selection
from sklearn import feature_extraction

import mlpy
from nltk.corpus import treebank
import random
import nltk, re, pprint
#from urllib.request import urlopen
from nltk.tokenize import StanfordSegmenter


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
    f = open('ENstopwords.txt', 'r')
    stopword = f.readlines()
    f.close()

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
                if not(word in stopword):
                    newsen = newsen + word + ' '

        rt.append(newsen.strip())

    return rt

def get_features(flst, sens):
    rt = []
    n = len(flst)
    farr = flst.tolist()
    stopword = []
    for sen in sens:
        a = [0] * n
        words = sen.split()
        for i in words:
            if i in farr:
                a[farr.index(i)] += 1
        rt.append(a)
    return np.array(rt)

def get_wordlist(sens):
    d = defaultdict()
    l = [] # all word list for return
    rtsens = [] # trimed sentences list for return
    for sen in sens:

        tokens = nltk.word_tokenize(sen)
        # 删除词list中的单独符号，比如','
        words = [word for word in tokens if not word in english_punctuations]
        # 删除词中的符号
        for word in words:
            for j in english_punctuations:
                if j in word:
                    word.replace(j, '')
        # 去掉停用词，其它词组成返回词表
        for word in words:
            if word in english_stopwords:
                if d[word] == 0:
                    d[word] += 1
                    l.append(word)
                else:
                    d[word] += 1
        rtsen = ''
        for word in words:
            rtsen = rtsen + word + ' '
        rtsens.append(rtsen.strip())

    return l, rtsens


if __name__ == "__main__":
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    a = open('ENstopwords.txt', 'r')
    english_stopwords = a.readlines()
    a.close()


    d0 = pd.read_csv("Youtube.csv")
    print d0.columns
    d0.drop('COMMENT_ID', axis=1)
    d0.drop('AUTHOR', axis=1)
    d0.drop('DATE', axis=1)
    d0.drop('CONTENT', axis=1)

    f = open('allsensclean.txt', 'r')
    sens = (f.readlines())
    f.close()
    d0['CONTENTS'] = pd.Series(sens)


    lk = tw.hasLink(d0["CONTENT"])
    d0['LINK'] = pd.Series(lk)

    print d0.head()

    wordlist, rtsens = get_wordlist(sens)
    d0['CLEAN'] = pd.Series(rtsens)
    word_features = get_features(wordlist, rtsens) # wordlist不含停用词， rtsens含停用词
    print word_features


    '''

    #frame = Frame2(10, 'allsensFrame.txt')
    # d0 -- all of the data

    se = pd.read_csv('ann.csv', header = None)
    se = open('ann.csv', 'r').readlines()

    d0['STANFORD'] = pd.Series(se)


    # d1 -- the data without links
    d1 = d0[d0['LINK'] == 0]
    d2 = d0[d0['LINK'] == 1]


    seed = random.randint(1,1000000)
    x_train, x_test, y_train, y_test = train_test_split(d1[['CONTENT', 'CLASS', 'STANFORD']], d1['CLASS'], test_size=0.2, random_state=seed)

    '''
    '''

    ########## features found finished! #############
    se = np.array(x_train['STANFORD'])
    li = []
    for line in se:
        a = line.split(',')
        b = [float(ii) for ii in a]
        li.append(b)

    xtrain2 = np.array(li)

    xtrain1 = get_features(word_features, x_train['CONTENT'])
    print len(xtrain1)
    print len(xtrain2)
    #xtrain2 = get_features(frame_features, x_train['FRAME'])
    xtrain = np.concatenate((xtrain1, xtrain2), axis=1)

    se = np.array(x_test['STANFORD'])
    li = []
    for line in se:
        a = line.split(',')
        b = [float(ii) for ii in a]
        li.append(b)

    xtest2 = np.array(li)



    xtest1 = get_features(word_features, x_test['CONTENT'])
    #xtest2 = get_features(frame_features, x_test['FRAME'])
    xtest = np.concatenate((xtest1, xtest2), axis=1)
    #xtest = xtest1
    '''
    pca = PCA(n_components=50)
    pca.fit(xtrain)
    xtrain = pca.transform(xtrain)
    xtest = pca.transform(xtest)

    pca1 = PCA(n_components=50)
    pca1.fit(xtrain1)
    xtrain1 = pca1.transform(xtrain1)
    xtest1 = pca.transform(xtest1)
    '''
    ytrain = np.array(y_train.tolist())
    ytest = np.array(y_test.tolist())
    #print xtrain


    #clf = svm.SVC()
    #clf = GaussianNB()
    #clf = BernoulliNB()
    #clf = tree.DecisionTreeClassifier()
    #clf = KNeighborsClassifier()
    clf = AdaBoostClassifier(n_estimators=100)
    #clf = MLPClassifier()

    #from stacked.stacked_generalization.lib.stacking import StackedClassifier
    #bclf = KNeighborsClassifier()
    #clfs = [GaussianNB(), BernoulliNB(), tree.DecisionTreeClassifier()]
    #clf = StackedClassifier(bclf, clfs)

    classifier = clf.fit(xtrain, ytrain)
    ypred = classifier.predict(xtest)
    A, P, R, F = ev.outcome(ypred, ytest)


    print "___________________________________________________________"
    classifier = clf.fit(xtrain1, ytrain)
    ypred1 = classifier.predict(xtest1)
    a, p, r, f = ev.outcome(ypred1, ytest)


    if (A > a):
        print "VVVVVVVVVVVVV"
        c += 1
    else:
        print "XXXXXXXXXXXXX"

    print "***********************************************************"
    print str(c) + u'/' + str(i+1)
    '''
    '''
    print (metrics.classification_report(ytest, ypred))
    print (metrics.confusion_matrix(ytest, ypred))
    print (metrics.classification_report(ytest, ypred1))
    print (metrics.confusion_matrix(ytest, ypred1))
    '''









