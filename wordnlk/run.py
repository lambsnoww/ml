#_*_coding:utf-8_*_

#仅利用语义信息进行机器学习模型训练，最高正确率达83%左右

import nltk
from nltk.probability import FreqDist
import pandas as pd
import numpy as np
import tools.wordProcess as tw
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



d0 = pd.read_csv("Youtube.csv")

lk = tw.hasLink(d0["CONTENT"])
ls = pd.DataFrame(lk)
#frame = Frame2(10, 'allsensFrame.txt')
# d0 -- all of the data
#d0 = pd.DataFrame({"CONTENT": content, "FRAME": frame.framelist, "CLASS": d0["CLASS"], "LINK": ls})

f = open('allsensclean.txt', 'r')
content = f.readlines()
#content = trimSens(list(d0['CONTENT']))

#d0 = pd.DataFrame({"CONTENT": content, "CLASS": d0["CLASS"], "LINK": ls})
# d1 -- the data without links
#d1 = d0[d0['LINK'] == 0]
#d2 = d0[d0['LINK'] == 1]
#yp = np.array(d2['LINK'])
#yo = np.array(d2['CLASS'])

#content_list = d1['CONTENT'].tolist()

se = pd.read_csv('sem.csv', header = None)
#s2 = d0['LINK']
#s3 = pd.read_csv('seminfo.csv', header = None)
s = pd.concat([se,ls], axis=1)
# annotated; VB before NP; Link
print s

x = s.values
y = np.array(d0['CLASS'])

seed = random.randint(1, 1000)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

# clf = svm.SVC()
# clf = GaussianNB()
#clf = BernoulliNB()
#clf = tree.DecisionTreeClassifier()
#clf = KNeighborsClassifier()
clf = AdaBoostClassifier(n_estimators=100)


#from stacked.stacked_generalization.lib.stacking import StackedClassifier
#bclf = KNeighborsClassifier()
#clfs = [GaussianNB(), BernoulliNB(), tree.DecisionTreeClassifier()]
#clf = StackedClassifier(bclf, clfs)

classifier = clf.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
A, P, R, F = ev.outcome(y_pred, y_test)

print "___________________________________________________________"




















#seed = random.randint(1,1000000)
#x_train, x_test, y_train, y_test = train_test_split(d1[['CONTENT', 'FRAME', 'CLASS']], d1['CLASS'], test_size=0.2, random_state=seed)
