#_*_coding:utf-8_*_
#运用sklearn.feature_extraction.text中的工具产生VSM向量空间模型
#降维后达89%左右
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import random
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
# the stacked classifier
from stacked.stacked_generalization.lib.stacking import StackedClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn import preprocessing
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
import tools.evaluate as ev
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA

import json

class Frame(object):
    # m - max number of frames for every sentences
    def __init__(self, m, filename):
        f = open(filename)

        ls = []
        ls2 = []
        arr = f.readlines()

        ss = []
        for line in arr:
            js = json.loads(line)
            #n = min(len(js['frames']), m)
            n = len(js['frames'])
            l = []
            l2 = []
            for i in range(n):
                str = js['frames'][i]['target']['name'].encode("gbk")
                score = js['frames'][i]['annotationSets'][0]['score']
                    #.encode("gbk")
                l.append(str)
                l2.append([str, score])
                #print str + '%f'%score
            l2 = sorted(l2, key=lambda x:x[1], reverse=True)
            ls.append(l)
            ls2.append(l2)

            ss = []
            for i in ls2:
                s = ''
                n = 0
                for j in i:
                    s = s + j[0] + ' '
                    n = n + 1
                    # print n
                    if n >= m:
                        break
                ss.append(s.strip())

        self.framelist = ss

def writeFrames():
    f = open('frame_info.txt', 'w')
    frame = Frame(10, 'allsensFrame.txt')



def hasLink(sens):
    #attLink = ""
    ls = []
    for sen in sens:
        l = 0
        if ("http://" in sen) or ("https://" in sen) or ("www." in sen) or (".com" in sen):
            l = 1
        elif sen.count('/') >= 3:
            l = 1
        # attLink = attLink + ',' + str(l)
        ls.append(l)

    return ls


def abnormal(sen):
    lensum = 0
    count = 0
    for word in sen.split():
        lensum += len(word)
        count += 1
    average = float(lensum) / count
    if average < 2:
        return True
    elif average > 15:
        return True
    else:
        return False


def trimSens():
    f = open('allsensclean.txt', 'r')
    p = open('clean.txt', 'w')
    sens = f.readlines()
    d = defaultdict()
    wordlist = []
    for sen in sens:
        words = sen.split()
        for word in words:
            word = word.lower()
            for i in english_punctuations:
                if i in word:
                    word.replace(i, '')
            if word in d:
                d[word] += 1
            else:
                d[word] = 1
                wordlist.append(word)










if __name__ == '__main__':
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\n', '<', '>', '~', '-', '_']

    f = open('allsensclean.txt', 'r')
    sens = f.readlines()
    f.close()
    d = pd.read_csv('YouTube.csv')
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sens)
    word = vectorizer.get_feature_names()
    print word
    x = X.toarray()
    y = np.array(d['CLASS'])

    pca = PCA(n_components=30)
    pca.fit(x)
    x_new = pca.transform(x)
    #plt.scatter(x_new[:, 0], x_new[:, 1], x_new[:, 2], marker='o')
    #plt.show()
    print x_new





    seed = random.randint(1, 10000)
    x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.3, random_state=seed)

    clf = svm.SVC()
    #clf = GaussianNB()
    #clf = BernoulliNB()
    #clf = tree.DecisionTreeClassifier()
    #clf = KNeighborsClassifier()
    #clf = AdaBoostClassifier(n_estimators=100)
    # clf = MLPClassifier()

    #from stacked.stacked_generalization.lib.stacking import StackedClassifier
    #bclf = KNeighborsClassifier()
    #clfs = [GaussianNB(), BernoulliNB(), tree.DecisionTreeClassifier()]
    #clf = StackedClassifier(bclf, clfs)

    classifier = clf.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    A, P, R, F = ev.outcome(y_pred, y_test)










