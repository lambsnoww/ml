#_*_coding:utf-8_*_
# semantic info - v n pos; sem vector; verb base list, verb base count
# run in this order: 1.cc 2.model


from sklearn.feature_extraction.text import CountVectorizer
from pycorenlp import StanfordCoreNLP
import tools
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from random import randint

if __name__ == '__main__':

    f = open('sens_xlink.txt', 'r')
    sens = f.readlines()
    f.close()

    f = open('labels.txt', 'r')
    labels = f.read().split()
    # do not forget to transfer chr to int
    labels = [ord(i)-ord('0') for i in labels]
    f.close()

    #sens and labels prepared

    lst, vec, verblist, countlemma = tools.get_sem_sequence_vector(sens)

    # verb vector: x_verb
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(verblist)
    word = vectorizer.get_feature_names()
    print word
    x_verb = X.toarray()

    #countlemma - 1-dimensional list


    seqs = tools.to_sequence(lst)
    #seqs:the sem info

    f = open('seqs.txt', 'w')
    for seq in seqs:
        f.write(seq + '\n')
    f.close()

    feature_list = []
    # h - NP o - VP

    features = []
    for seq in seqs:
        t = seq.split(',')
        count = 0
        count2 = 0
        for i in t:
            if i.find('h') >= 0 and i.find('o') >= 0:
                if i.find('h') > i.find('o'):
                    count += 1
            elif i.find('o') >= 0 and i.find('h') == -1:
                count2 += 1
        features.append([count, count2])
    #print features

    #vec and features

    countlemma = np.array(countlemma).reshape(-1,1)

    feat = np.concatenate((vec, features, x_verb, countlemma), axis=1)

    print x_verb[:5]
    print countlemma[:5]



    f = open('sem_features.txt', 'w')
    for i in feat:
        s = ''
        for j in i:
            s += (',' + str(j))
        s = s[1:]
        f.write(s + '\n')
    f.close()


    x = feat
    y = labels
    seed = randint(1, 10000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    tools.evaluate(y_pred, y_test)















