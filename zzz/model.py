#_*_coding:utf-8_*_

from sklearn import svm
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

    clf = svm.SVC()
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





