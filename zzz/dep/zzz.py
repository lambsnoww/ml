#_*_coding:utf-8_*_
#combining knn and other features
from pycorenlp import StanfordCoreNLP
import Levenshtein
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import evaluate
from sklearn import metrics
import math
from collections import defaultdict
from sklearn.metrics import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

def gaussian(dist, a=1, b=0, c=3):
    return a*math.e**(-(dist-b)**2/(2*c**2))
def knn_lin0(k, x_train, y_train, test, aa, bb, cc):
    a = test.split(',')
    a = [i for i in a if i != '']

    dis_list = []
    for index in range(len(x_train)):
        dis = 9999
        b = x_train[index].split(',')
        b = [i for i in b if i != '']
        for i in a:
            for j in b:
                tmp = Levenshtein.distance(i, j)
                if dis > tmp:
                    dis = tmp
        dis_list.append((dis, index))
    nearest = sorted(dis_list, key=lambda x: x[0])
    #print nearest[:100]
    if k < len(nearest):
        ne = nearest[:k]
    else:
        ne = nearest

    summ = 0
    for i in ne:
        summ += y_train[i[1]] * gaussian(i[0], aa, bb, cc)
    if summ * 2 >= len(ne):
        return 1
    else:
        return 0

def knn_lin1(k, x_train, y_train, test):
    a = test.split(',')
    a = [i for i in a if i != '']
    dis_list = []
    for index in range(len(x_train)):
        dis = 9999
        b = x_train[index].split(',')
        b = [i for i in b if i != '']
        for i in a:
            for j in b:
                tmp = Levenshtein.distance(i, j)
                if dis > tmp:
                    dis = tmp
        dis_list.append((dis, index))
    nearest = sorted(dis_list, key=lambda x: x[0])
    #print nearest[:100]
    if k < len(nearest):
        ne = nearest[:k]
    else:
        ne = nearest

    summ = 0
    for i in ne:
        summ += y_train[i[1]] * gaussian(i[0])
    if summ * 2 >= len(ne):
        return 1
    else:
        return 0
def run_pred():
    seed = random.randint(1, 100000)
    x_train, x_test, y_train, y_test = train_test_split(seqs, labels, test_size=0.7, random_state=seed)

    k = 35
    aa = 1; bb = 0; cc = 3

    y_pred0 = []
    for test in x_test:
        y_pred0.append(knn_lin0(k, x_train, y_train, test, aa, bb, cc))
    evaluate.outcome(y_pred0, y_test)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sens)
    word = vectorizer.get_feature_names()
    #print word
    x_word = X.toarray()
    print x_word
    x_train_word, x_test_word, y_train, y_test = train_test_split(x_word, labels, test_size=0.7, random_state=seed)

    clf = MLPClassifier()
    clf.fit(x_train_word, y_train)
    y_pred_word = clf.predict(x_test_word)

    y_pred1 = [-1] * len(x_test)
    for i in range(len(x_test)):
        test = x_test[i]
        out = knn_lin1(k, x_train, y_train, test)
        if out == 1:
            y_pred1[i] = 1
        else:
            y_pred1[i] = y_pred_word[i]
    evaluate.outcome(y_pred1, y_test)
    evaluate.outcome(y_pred_word, y_test)

    sens_train, sens_test, x, y = train_test_split(sens, labels, test_size=0.7, random_state=seed)
    for i in range(len(y_pred_word)):
        if (y_pred_word[i] != y_test[i]):
            print y_pred_word[i], y_test[i]
            print sens_test[i]


if __name__ == '__main__':

    f = open('labels.txt', 'r')
    tmp_labels = f.read().split()
    f.close()
    labels = [ord(i) - ord('0') for i in tmp_labels]

    f = open('sem_seq.txt', 'r')
    seqs = f.readlines()
    f.close()

    f = open('sens_xlink.txt', 'r')
    sens = f.readlines()
    f.close()



    df = pd.DataFrame()
    df['SENS'] = pd.Series(sens)
    df['CLASS'] = pd.Series(labels)

    spam = df[df['CLASS'] == 1]
    comm = df[df['CLASS'] == 0]

    #print spam.head(50)
    #print comm.head(50)
    run_pred()


