#_*_coding:utf-8_*_

#协同训练方法

import nltk
from nltk.probability import FreqDist
import pandas as pd
import numpy as np
#import tools.wordProcess as tw
from wordAttrExtraction.Word import *
from wordAttrExtraction.Frame import *
from wordAttrExtraction.Frame2 import *
from tools import *

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

#import tools.evaluate as ra
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

import mlpy
import random
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.lda import LDA
from sklearn.neural_network import MLPClassifier
#import gensim
from gensim.models.keyedvectors import KeyedVectors
import xue.tool as tool
import re
import hmmlearn
from model_svm_tools import *


def main():

    x_sem, y = get_sem_features()
    f = open('sens_xlink.txt', 'r'); sens = f.readlines(); f.close()

    seed = randint(1, 10000)
    x_train_sem, x_test_sem, y_train, y_test = train_test_split(x_sem, y, test_size=0.2, random_state=seed)
    x_train_sens, x_test_sens, tmp1, tmp2 = train_test_split(sens, y, test_size=0.2, random_state=seed)


    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sens)
    word = vectorizer.get_feature_names()
    #print word
    x_word = X.toarray()
    #x_origin = x_word

    pca = PCA(n_components=200)
    pca.fit(x_word)
    x_word = pca.transform(x_word)

    seed = random.randint(1, 1000)
    x_train_word, x_test_word, y_train, y_test = train_test_split(x_word, y, test_size=0.2, random_state=seed)
    #x_train_sem, x_test_sem, y_train, y_test = train_test_split(x_sem, y, test_size=0.2, random_state=seed)

    nt = len(x_train_word)
    r = 0.5 # the ratio of labeled data / unlabeled data
    rr = int(r * nt)
    sample_train_L1 = x_train_word[0:rr]
    sample_train_L2 = x_train_sem[0:rr]
    yL = y_train[0:rr]
    sample_train_U1 = x_train_word[rr:]
    sample_train_U2 = x_train_sem[rr:]
    yU = y_train[rr:]
    # traditional mathods
    clf3 = AdaBoostClassifier()
    clf3.fit(sample_train_L1, yL)
    y_pred3 = clf3.predict(x_test_word)
    clf4 = svm.SVC()
    clf4.fit(sample_train_L2, yL)
    y_pred4 = clf4.predict(x_test_sem)
    tools.evaluate(y_pred3, y_test)
    tools.evaluate(y_pred4, y_test)
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    loops = 30

    for i in range(loops):
        print i
        if len(sample_train_U1) == 0:
            break
        #clf = svm.SVC()
        #clf = GaussianNB()
        #clf = KNeighborsClassifier()
        clf = AdaBoostClassifier(n_estimators=100)
        clf.fit(sample_train_L1, yL)
        # clf = svm.SVC()
        # clf = BernoulliNB()
        # clf = tree.DecisionTreeClassifier()
        # clf = KNeighborsClassifier()
        #clf = AdaBoostClassifier(n_estimators=100)
        # from stacked.stacked_generalization.lib.stacking import StackedClassifier
        # bclf = KNeighborsClassifier()
        # clfs = [GaussianNB(), BernoulliNB(), tree.DecisionTreeClassifier()]
        # clf = StackedClassifier(bclf, clfs)
        #clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)
        clf2 = svm.SVC(probability=True)
        clf2.fit(sample_train_L2, yL)
        yU_pred1 = clf.predict(sample_train_U1)
        proba1 = clf.predict_proba(sample_train_U1)
        yU_pred1_proba = []

        for i in range(len(proba1)):
            if yU_pred1[i] == 0:
                yU_pred1_proba.append(proba1[i][0])
            else:
                yU_pred1_proba.append(proba1[i][1])

        yU_pred1_proba = np.array(yU_pred1_proba)

        ind = (-yU_pred1_proba).argsort()
        yU_pred2 = clf2.predict(sample_train_U2)
        proba2 = clf2.predict_proba(sample_train_U2)
        yU_pred2_proba = []
        for i in range(len(proba2)):
            if yU_pred1[i] == 0:
                yU_pred2_proba.append(proba2[i][0])
            else:
                yU_pred2_proba.append(proba2[i][1])
        yU_pred2_proba = np.array(yU_pred2_proba)

        ind2 = (-yU_pred2_proba).argsort()
        p = 10
        n = 10
        cnt = 1000
        del_list = []
        while p+n > 0:
            cnt -= 1
            if cnt == 0:
                break
            if len(sample_train_U1) == 0:
                break
            #a = random.randint(0, len(sample_train_U1)-1)
            for a in ind:
                if (yU_pred1[a] == 1 and p > 0) or \
                        (yU_pred1[a] == 0 and n > 0):
                    t = yU_pred1[a]
                    tmp = sample_train_U1[a]
                    tmp2 = sample_train_U2[a]
                    #add
                    sample_train_L1 = np.row_stack((sample_train_L1, tmp))
                    sample_train_L2 = np.row_stack((sample_train_L2, tmp2))
                    yL = np.append(yL, t)
                    #delete
                    del_list.append(a)

                    #sample_train_U1 = np.delete(sample_train_U1, a, axis=0)
                    #sample_train_U2 = np.delete(sample_train_U2, a, axis=0)
                    #yU = np.append(yU[0:a], yU[a+1:]) # ndarray删除
                    if t == 1:
                        p -= 1
                    elif t == 0:
                        n -= 1
        p = 10
        n = 10
        cnt = 1000
        while p+n > 0:
            cnt -= 1
            if cnt == 0:
                break
            if len(sample_train_U1) == 0:
                break
            #a = random.randint(0, len(sample_train_U1)-1)
            for a in ind2:
                if (yU_pred2[a] == 1 and p > 0) or \
                        (yU_pred2[a] == 0 and n > 0) and \
                                (a not in del_list):
                    t = yU_pred2[a]
                    tmp = sample_train_U1[a]
                    tmp2 = sample_train_U2[a]
                    #add
                    sample_train_L1 = np.row_stack((sample_train_L1, tmp))
                    sample_train_L2 = np.row_stack((sample_train_L2, tmp2))
                    yL = np.append(yL, t)
                    #delete
                    del_list.append(a)
                    #sample_train_U1 = np.delete(sample_train_U1, a, axis=0)
                    #sample_train_U2 = np.delete(sample_train_U2, a, axis=0)
                    #yU = np.append(yU[0:a], yU[a+1:]) # ndarray删除
                    if t == 1:
                        p -= 1
                    elif t == 0:
                        n -= 1
        sample_train_U1 = np.delete(sample_train_U1, del_list, axis=0)
        sample_train_U2 = np.delete(sample_train_U2, del_list, axis=0)
        yU = np.delete(yU, del_list, axis=0)
        #yU = np.append(yU[0:a], yU[a+1:]) # ndarray删除


    y_pred1 = clf.predict(x_test_word)
    yp1 = clf.predict_proba(x_test_word)
    y_pred2 = clf2.predict(x_test_sem)
    yp2 = clf.predict_proba(x_test_word)

    y_pred = []
    for i in range(len(y_pred1)):
        if y_pred1[i] == y_pred2[i]:
            y_pred.append(y_pred1[i])
        else:
            if yp1[i][y_pred1[i]] > yp2[i][y_pred2[i]]:
                y_pred.append(y_pred1[i])
            else:
                y_pred.append(y_pred2[i])




    tools.evaluate(y_pred1, y_test)
    tools.evaluate(y_pred2, y_test)
    tools.evaluate(y_pred, y_test)















    '''

    # annotated; VB before NP; Link
    #print s

    #x_sem = s.drop('CLASS', axis=1).values
    #x_scaled = preprocessing.scale(x)
    #x = x_scaled
    #print x

    x_all = np.concatenate((x_word, x_sem), axis=1)

    #b1 = 0; e1 = 349; b2 = 350; e2 = 699; b3 = 700; e3 = 1137;
    #b4 = 1138; e4 = 1584; b5 = 1585; e5 = 1954

    #x_all = x_all[b1:b2,]; x_sem = x_sem[b1:b2,]; x_word = x_word[b1:b2,]; y = y[b1:b2,]
    #x_all = x_all[b2:b3,]; x_sem = x_sem[b2:b3,]; x_word = x_word[b2:b3,]; y = y[b2:b3,]
    #x_all = x_all[b3:b4,]; x_sem = x_sem[b3:b4,]; x_word = x_word[b3:b4,]; y = y[b3:b4,]
    #x_all = x_all[b4:b5,]; x_sem = x_sem[b4:b5,]; x_word = x_word[b4:b5,]; y = y[b4:b5,]
    #x_all = x_all[b5:e5+1,]; x_sem = x_sem[b5:e5+1,]; x_word = x_word[b5:e5+1,]; y = y[b5:e5+1,]

    label = 'all-adaboost'
    seed = random.randint(1, 1000)
    x_train, x_test, y_train, y_test = train_test_split(x_all, y, test_size=0.2, random_state=seed)

    #clf = svm.SVC()
    #clf = GaussianNB()
    #clf = BernoulliNB()
    #clf = tree.DecisionTreeClassifier()
    #clf = KNeighborsClassifier()
    clf = AdaBoostClassifier(n_estimators=100)
    #from stacked.stacked_generalization.lib.stacking import StackedClassifier
    #bclf = KNeighborsClassifier()
    #clfs = [GaussianNB(), BernoulliNB(), tree.DecisionTreeClassifier()]
    #clf = StackedClassifier(bclf, clfs)

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)


    label2 = 'sem-adaboost'
    x_train, x_test, y_train, y_test = train_test_split(x_sem, y, test_size=0.2, random_state=seed)
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(x_train, y_train)
    y_pred2 = clf.predict(x_test)

    label3 = 'sem-ida'
    lda = LinearDiscriminantAnalysis(n_components=1)
    classifier = lda.fit(x_train, y_train)
    y_pred3 = classifier.predict(x_test)

    label4 = 'sem-KNN'
    clf = KNeighborsClassifier(100)
    clf.fit(x_train, y_train)
    y_pred4 = clf.predict(x_test)

    label5 = 'sem-svm'
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred5 = clf.predict(x_test)

    label6 = 'word-adaboost'
    x_train, x_test, y_train, y_test = train_test_split(x_word, y, test_size=0.2, random_state=seed)
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(x_train, y_train)
    y_pred6 = clf.predict(x_test)




    #A, P, R, F = ev.outcome(y_pred, y_test)
    #A, P, R, F = ev.outcome(y_pred2, y_test)
    #A, P, R, F = ev.outcome(y_pred3, y_test)
    label7 = 'word-MLP'
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)
    clf.fit(x_train, y_train)
    y_pred7 = clf.predict(x_test)

    print label
    A, P, R, F = ev.outcome(y_pred, y_test)
    print label2
    A, P, R, F = ev.outcome(y_pred2, y_test)
    print label3
    A, P, R, F = ev.outcome(y_pred3, y_test)
    print label4
    A, P, R, F = ev.outcome(y_pred4, y_test)
    print label5
    A, P, R, F = ev.outcome(y_pred5, y_test)
    print label6
    A, P, R, F = ev.outcome(y_pred6, y_test)
    print label7
    A, P, R, F = ev.outcome(y_pred7, y_test)

    print "___________________________________________________________"


    #SVM, KNN, Adaboost performs better

    #all sem word
    ##################################

    dataframe = pd.DataFrame(x_sem)
    dataframe['CLASS'] = pd.Series(y)

    dataframe['LINK'] = ls
    dataframe = dataframe[dataframe['LINK']==0]
    x = dataframe.drop('CLASS',axis=1).values
    em = pd.read_csv('em.csv', header=None)
    x = np.concatenate((x,em.values), axis=1)
    #x = em
    y = np.array(dataframe['CLASS'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

    #clf = svm.SVC()
    #clf = GaussianNB()
    #clf = BernoulliNB()
    #clf = tree.DecisionTreeClassifier()
    #clf = KNeighborsClassifier()
    clf = AdaBoostClassifier(n_estimators=100)
    #from stacked.stacked_generalization.lib.stacking import StackedClassifier
    #bclf = KNeighborsClassifier()
    #clfs = [GaussianNB(), BernoulliNB(), tree.DecisionTreeClassifier()]
    #clf = StackedClassifier(bclf, clfs)

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print "em-added"
    A, P, R, F = ev.outcome(y_pred, y_test)
    '''





if __name__ == '__main__':

    main()












