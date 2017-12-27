#_*_coding:utf-8_*_

import numpy as np
import pandas as pd
import Levenshtein
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from random import randint
import evaluate

def knn(k, x_train, y_train, test):
    dist = []
    for i in x_train:
        #print i
        #print test
        dist.append(Levenshtein.distance(i, test))
    dist = np.array(dist)

    ind = (-dist).argsort()

    if len(dist) < k:
        k = len(dist)

    sum = 0
    for i in ind[0:k]:
        sum += y_train[i]

    if (sum * 2 >= k):
        return 1
    else:
        return 0

if __name__ == '__main__':


    f = open('labels.txt', 'r')
    labels = f.read().split()
    f.close()
    labels = [ord(i)-ord('0') for i in labels]
    f = open('sens_xlink.txt', 'r')
    sens = f.readlines()
    f.close()
    # sens labels prepared
    seed = randint(1, 1000)
    x_train, x_test, y_train, y_test = train_test_split(sens, labels, test_size=0.2, random_state=seed)

    k = 15
    y_pred = [0] * len(x_test)
    for i in range(len(x_test)):
        y_pred[i] = knn(k, x_train, y_train, x_test[i])

    acc = accuracy_score(y_test, y_pred)
    pr = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f = f1_score(y_test, y_pred)
    print 'accuracy, precision, recall, f-measure'
    print acc, pr, rec, f
    evaluate.outcome(y_pred, y_test)







