#_*_coding:utf-8_*_

import pandas as pd
import numpy as np
import tools.wordProcess as tw
from Word import *
from Frame import *
from Frame2 import *
import tools.evaluate as ev

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
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import string
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import copy
from stacked.stacked_generalization.lib.stacking import StackedClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer





if __name__ == "__main__":

    # 1 - frame and word
    # 0 - word only
    # -1 - frame only
    frameOrNot = 1

    #method = "SVM"
    #method = "GaussianNB"
    #method = "BernoulliNB"
    #method = "Decision Tree"
    method = "AdaBoost"
    #method = "KNN"

    #d = pd.read_csv("Youtube01-Psy.csv")
    #d = pd.read_csv("Youtube02-KatyPerry.csv")
    #d = pd.read_csv("Youtube03-LMFAO.csv")
    #d = pd.read_csv("Youtube04-Eminem.csv")
    #d = pd.read_csv("Youtube05-Shakira.csv")
    d0 = pd.read_csv("Youtube.csv")
    f = open('allsensFrame.txt', 'r')
    f0 = pd.Series(f.readlines())

    lk = tw.hasLink(d0["CONTENT"])
    ls = pd.Series(lk)
    d0 = pd.DataFrame({"CONTENT": d0["CONTENT"], "FRAME": f0, "CLASS": d0["CLASS"], "LINK": ls})
    frame = Frame2(100, 'allsensFrame.txt')
    print frame.framelist
















