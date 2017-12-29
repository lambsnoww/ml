#_*_coding:utf-8_*_
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import *
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# the stacked classifier
from stacked.stacked_generalization.lib.stacking import StackedClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tools
from random import randint
from sklearn.feature_extraction.text import CountVectorizer

def get_sem_features():
    df = pd.read_csv('sem_features.txt', header=None)

    f = open('labels.txt', 'r')
    labels = f.read().split()
    # do not forget to transfer chr to int
    labels = [ord(i) - ord('0') for i in labels]
    f.close()

    f = open('sens_xlink.txt', 'r')
    sens = f.readlines()
    f.close()

    notwordcount = []
    for text in sens:
        a = text.split()
        senlen = len(a)
        nc = 0
        b = 0
        for word in a:
            tmp = tools.wordTrim(word).lower()
            if len(tmp) == 0:
                continue
            if not tools.wordnet.synsets(tmp):
                nc += 1
        if senlen != 0:
            b = float(nc) / senlen
        notwordcount.append([nc, b])

    abn = tools.abnormal(sens)
    # abn = np.array(abn).reshape(-1, 2)

    word_features = tools.get_word_feature(sens)

    x = df.values
    x = np.concatenate((x, notwordcount, abn, word_features), axis=1)
    y = labels
    #print len(x[0])
    #print len(x)
    return x, y
    #seed = randint(1, 10000)
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    #x_train_sens, x_test_sens, y_train, y_test = train_test_split(sens, y, test_size=0.2, random_state=seed)





def get_word_features():

    f = open('sens_xlink.txt', 'r'); sens = f.readlines(); f.close()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sens)
    word = vectorizer.get_feature_names()
    print word
    x_word = X.toarray()
    #x_origin = x_word

    #pca = PCA(n_components=200)
    #pca.fit(x_word)
    #x_word = pca.transform(x_word)


    f = open('labels.txt', 'r')
    labels = f.read().split()
    # do not forget to transfer chr to int
    labels = [ord(i) - ord('0') for i in labels]
    f.close()

    f = open('sens_xlink.txt', 'r')
    sens = f.readlines()
    f.close()

    notwordcount = []
    for text in sens:
        a = text.split()
        senlen = len(a)
        nc = 0
        b = 0
        for word in a:
            tmp = tools.wordTrim(word).lower()
            if len(tmp) == 0:
                continue
            if not tools.wordnet.synsets(tmp):
                nc += 1
        if senlen != 0:
            b = float(nc) / senlen
        notwordcount.append([nc, b])

    abn = tools.abnormal(sens)
    # abn = np.array(abn).reshape(-1, 2)

    word_features = tools.get_word_feature(sens)

    x = np.concatenate((x_word, notwordcount, abn, word_features), axis=1)
    y = labels

    return x, y
    # seed = randint(1, 10000)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    # x_train_sens, x_test_sens, y_train, y_test = train_test_split(sens, y, test_size=0.2, random_state=seed)







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

    notwordcount = []
    for text in sens:
        a = text.split()
        senlen = len(a)
        nc = 0
        b = 0
        for word in a:
            tmp = tools.wordTrim(word).lower()
            if len(tmp) == 0:
                continue
            if not tools.wordnet.synsets(tmp):
                nc += 1
        if senlen != 0:
            b = float(nc) / senlen
        notwordcount.append([nc, b])

    abn = tools.abnormal(sens)
    #abn = np.array(abn).reshape(-1, 2)

    word_features = tools.get_word_feature(sens)

    x = df.values
    x = np.concatenate((x, notwordcount, abn, word_features), axis=1)
    y = labels

    seed = randint(1, 10000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_train_sens, x_test_sens, y_train, y_test = train_test_split(sens, y, test_size=0.2, random_state=seed)

    # MLP(10)和SVM(001,100)
    # clf = MLPClassifier(hidden_layer_sizes=10)
    clf = svm.SVC(gamma=0.001, C=100)
    #clf = svm.SVC(gamma=0.00001, C=100) # precision
    #clf = LinearDiscriminantAnalysis(n_components=1)
    #clf = GaussianNB()
    #clf = BernoulliNB()
    #clf = tree.DecisionTreeClassifier()
    #clf = AdaBoostClassifier(n_estimators=100)
    #clf = KNeighborsClassifier()
    # clf = AdaBoostClassifier(n_estimators=100)

    #from stacked.stacked_generalization.lib.stacking import StackedClassifier
    #bclf = KNeighborsClassifier()
    #clfs = [GaussianNB(), BernoulliNB(), tree.DecisionTreeClassifier()]
    #clf = StackedClassifier(bclf, clfs)

    clf.fit(x_train, y_train)
    y_pred_train = clf.predict(x_train)
    y_pred = clf.predict(x_test)

    app = [1] * int(210*0.2)
    app2 = [1] * int(210*0.8)
    y_pred = np.concatenate((y_pred, app), axis=0)
    y_test = np.concatenate((y_test, app), axis=0)
    y_pred_train = np.concatenate((y_pred_train, app2), axis=0)
    y_train = np.concatenate((y_train, app2), axis=0)

    print "On train data:"
    tools.evaluate(y_pred_train, y_train)
    print "On test data:"

    tools.evaluate(y_pred, y_test)
    tools.print_misclassified(y_pred, y_test, x_test_sens)
    #tools.print_misclassified(y_pred_train, y_train, x_train_sens)






