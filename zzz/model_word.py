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

    seed = randint(1, 10000)
    x_train_sem, x_test_sem, y_train, y_test = train_test_split(x_sem, y, test_size=0.2, random_state=seed)

    x_word, tmp = get_word_features()


    #x_origin = x_word

    #pca = PCA(n_components=200)
    #pca.fit(x_word)
    #x_word = pca.transform(x_word)

    x_train_word, x_test_word, y_train, y_test = train_test_split(x_word, y, test_size=0.2, random_state=seed)
    #x_train_sens, x_test_sens, tmp1, tmp2 = train_test_split(sens, y, test_size=0.2, random_state=seed)
    #x_train_sem, x_test_sem, y_train, y_test = train_test_split(x_sem, y, test_size=0.2, random_state=seed)

    #clf = MLPClassifier(10)
    clf = svm.SVC()
    clf.fit(x_train_word, y_train)
    y_pred1 = clf.predict(x_test_word)

    clf2 = svm.SVC(gamma=0.001, C=100)
    clf2.fit(x_train_sem, y_train)
    y_pred2 = clf2.predict(x_test_sem)

    tools.evaluate(y_pred1, y_test)
    tools.evaluate(y_pred2, y_test)















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












