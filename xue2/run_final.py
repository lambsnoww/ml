#_*_coding:utf-8_*_

#仅利用语义信息进行机器学习模型训练，最高正确率达87%左右
#利用Fisher LDA判别，正确率87%左右

import nltk
from nltk.probability import FreqDist
import pandas as pd
import numpy as np
#import tools.wordProcess as tw
from wordAttrExtraction.Word import *
from wordAttrExtraction.Frame import *
from wordAttrExtraction.Frame2 import *
import tools.evaluate as ev
from wordAttrExtraction.run7 import *

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


def main():

    ab = pd.read_csv('abnormal.csv', header=None)

    d0 = pd.read_csv("Youtube.csv")
    lk = tw.hasLink(d0["CONTENT"])
    ls = pd.DataFrame(lk)

    f = open('sens_final.txt', 'r')
    sens = f.readlines()
    f.close()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sens)
    word = vectorizer.get_feature_names()
    print word
    x_word = X.toarray()
    x_origin = x_word

    pca = PCA(n_components=10)
    pca.fit(x_word)
    x_word = pca.transform(x_word)
    df_word = pd.DataFrame(x_word)
    df_word['LINK'] = ls

    f = open('emoji.txt', 'r')
    a = f.readlines()
    e = np.array([int(i) for i in a])
    f.close()

    df_word['EMOJI'] = pd.Series(e)
    df_word['AB0'] = ab[0]
    df_word['AB1'] = ab[1]

    #df_word = df_word[df_word['LINK'] == 0]
    x_word = df_word.values



    #__________________________semantic______


    se = pd.read_csv('sem_all.csv', header = None)
    df = pd.DataFrame(se)

    df['LINK'] = ls
    df['CLASS'] = d0['CLASS']
    df['EMOJI'] = pd.Series(e)
    df['AB0'] = ab[0]
    df['AB1'] = ab[1]
    #s = df[df['LINK'] == 0]
    y = np.array(df['CLASS'])
    df = df.drop('CLASS', axis=1)
    s = df.values
    attr = (tool.get_word_features())
    x_sem = np.concatenate((s, attr), axis=1)

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


def word_len():
    f = open('allsensclean.txt', 'r')
    sens = f.readlines()
    pattern = r'[,.:;?!~*]+'
    p = re.compile(pattern)


def trim_sens(infile, tofile): # 去乱符号并统一用句点代替，单词缩写改为全拼
    b = [',', '.', ':', ';', '?', '!']
    f = open(infile, 'r')
    sens = f.readlines()
    pattern = r'[,.:;?!~*]{2,}'
    re.compile(pattern)
    allsens = []
    for sen in sens:
        found = re.findall(pattern, sen)
        for i in found:
            tmp = i[0] + ' '
            sen = sen.replace(i, tmp)
        allsens.append(sen)
    sens = allsens


    abbd = defaultdict()
    abbd['ur'] = 'you are'
    abbd['u'] = 'you'
    abbd['im'] = 'i am'
    abbd['thx'] = 'thanks'
    abbd['plz'] = 'please'
    abbd['sub'] = 'subscribe'
    abbd['dont'] = 'don\'t'
    abbd['yr'] = 'year'

    fw = open(tofile, 'w')
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\n', '<', '>', '~', '_']
    a = ['@', '#', '$', '%', '^', '&', '*', '~']
    b = [',', '.', ':', ';', '?', '!']
    for sen in sens:
        ll = ''
        for word in sen.split():
            if word == '':
                continue
            tmp = clean(word.lower())
            if tmp == '':
                continue
            symbol = word[-1]
            if symbol in a:
                symbol = '.'
            if tmp in abbd:
                tmp = abbd[tmp]
            if symbol not in b:
                symbol = ''
            ll = ll + tmp + symbol + ' '
        fw.write(ll + '\n')
    fw.close

def clean(word):
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\n', '<', '>', '~', '_']
    add = ['-', "'"]
    for i in english_punctuations:
        if i in word:
            word = word.replace(i, '')
    return word


def has_emoji(infile, tofile):
    f = open('e.txt', 'r')
    es = f.read().splitlines()
    print es
    f.close()
    f = open(infile, 'r')
    sens = f.readlines()
    f.close()
    f2 = open(tofile, 'w')
    l = []
    for sen in sens:
        sen_new = ''
        c = 0
        words = sen.split()
        for e in es:
            if e in words:
                c += 1
                if words.index(e) > 0:
                    words[words.index(e) - 1] += '.'
                words.remove(e)
        for word in words:
            sen_new += word + ' '
        sen_new = sen_new.strip()
        f2.write(sen_new + '\n')
        l.append(c)
    f2.close()
    return l


if __name__ == '__main__':

    emoji = has_emoji('allsensclean.txt', 'sens_emoji.txt')
    #print emoji
    #f = open('emoji.txt', 'w')
    #for i in emoji:
    #    f.write(str(i) + '\n')
    #f.close()
    trim_sens('sens_emoji.txt', 'sens_final.txt')

    main()












