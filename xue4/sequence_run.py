#_*_coding:utf-8_*_

import Levenshtein
import pandas as pd
import numpy as np
import trim as tm
from sklearn.model_selection import train_test_split
import random
import evaluate
from sklearn.feature_extraction.text import CountVectorizer
#-------------------------
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


#多值比较只取最小值
def knn_lin(k, x_train, y_train, test):

    a = test.split(',')

    dis_list = []
    for index in range(len(x_train)):
        dis = 9999
        b = x_train[index].split(',')
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
        summ += y_train[i[1]]
    if summ * 2 >= len(ne):
        return 1
    else:
        return 0
#多值比较取平均值
def knn_lin2(k, x_train, y_train, test):
    a = test.split(',')
    dis_list = []
    for index in range(len(x_train)):
        dis = 0
        b = x_train[index].split(',')
        for i in a:
            for j in b:
                tmp = Levenshtein.distance(i, j)
                dis += tmp
        dis = float(dis)/(len(a) * len(b))
        dis_list.append((dis, index))
    nearest = sorted(dis_list, key=lambda x: x[0])
    if k < len(nearest):
        ne = nearest[:k]
    else:
        ne = nearest

    summ = 0
    for i in ne:
        summ += y_train[i[1]]
    if summ * 2 >= len(ne):
        return 1
    else:
        return 0
#copy of knn_lin2
def knn_lin3(k, x_train, y_train, test):
    a = test.split(',')
    dis_list = []
    for index in range(len(x_train)):
        dis = 0
        b = x_train[index].split(',')
        for i in a:
            for j in b:
                tmp = Levenshtein.distance(i, j)
                dis += tmp
        dis = float(dis)/(len(a) * len(b))
        dis_list.append((dis, index))
    nearest = sorted(dis_list, key=lambda x: x[0])

    if k < len(nearest):
        ne = nearest[:k]
    else:
        ne = nearest

    summ = 0
    for i in ne:
        summ += y_train[i[1]]
    if summ * 2 >= len(ne):
        return 1
    else:
        return 0

if __name__ == '__main__':

    f = open('seq_char.txt', 'r')
    lines = f.read().splitlines()
    f.close()

    d = pd.read_csv('YouTube.csv')
    labels = np.array(d['CLASS'])
    links = pd.Series(tm.hasLink(d['CONTENT']))

    df = pd.DataFrame({'CONTENT': pd.Series(lines), 'LINK': links, 'CLASS': labels})

    df['INDEX']=pd.Series(np.arange(0,1955))
    d = df[df['LINK'] == 0]
    print df[df['CLASS']==1]
    print "*****************************"
    print df[df['CLASS']==0]
   #---------------------------to csv----------
    f = open('sens_final.txt', 'r')
    sens = f.readlines()
    f.close()
    df['SENTENCES'] = pd.Series(sens)
    tmp = pd.DataFrame({'CLASS':df['CLASS'], 'CONTENT':df['SENTENCES']})
    tmp.insert(0,'LINK', df['LINK'])
    tmp.to_csv('DataFrame.csv')

    #——————————————data prepared——————————————

    x = d.drop('CLASS', axis=1)
    x = np.array(x['CONTENT'])
    y = np.array(d['CLASS'])
    seed = random.randint(1,10000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)





    y_pred = []
    for test in x_test:
        y_pred.append(knn_lin(50, x_train, y_train, test))
    y_pred2 = []
    for test in x_test:
        y_pred2.append(knn_lin2(50, x_train, y_train, test))

    #evaluate.outcome(y_pred, y_test)
    #evaluate.outcome(y_pred2, y_test)
    print metrics.confusion_matrix(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred2)


    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(x)
    word = vectorizer.get_feature_names()
    #print word
    x_word = X.toarray()
    print len(x_word[0])

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




    x_train, x_test, y_train, y_test = train_test_split(x_word, y, test_size=0.2, random_state=seed)
    clf.fit(x_train, y_train)
    y_pred3 = clf.predict(x_test)
    evaluate.outcome(y_pred3, y_test)
    print metrics.confusion_matrix(y_test, y_pred3)

