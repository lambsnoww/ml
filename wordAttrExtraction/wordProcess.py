#_*_coding:utf-8_*_
import pandas as pd
import numpy as np
import csv
import re
from collections import Counter
from collections import defaultdict
import tools.wordProcess as ts
import gensim

def hasLink(sens):
    #attLink = ""
    ls = []
    for sen in sens:
        l = 0
        if ("http://" in sen) or ("https://" in sen) or ("www." in sen) or (".com" in sen):
            l = 1
        # attLink = attLink + ',' + str(l)
        ls.append(l)

    return ls
#deprecated
def DEPRECATEDvectorSpace(sens):
    model = gensim.models.Word2Vec(sens, size=200, workers=4)

    f = open("ENstopwords.txt", 'r')
    sw = f.readlines()
    senCounts = []
    allCounts = defaultdict(int)
    for sen in sens:
        counts = defaultdict(int)
        c = 0
        for word in sen.split():
            word = ts.wordTrim(word)
            flag = True
            for w in sw:
                if word == w.strip():
                    flag = False
                    break
            if flag:
                c = c + 1
                counts[word] += 1
                allCounts[word] += 1
        if c == 0:
            # print sen
            continue
        for word in sen.split():
            word = ts.wordTrim(word)
            for w in sw:
                if word == w.strip():
                    flag = False
                    break
            if flag:
                counts[word] = counts[word] / c

        senCounts.append(counts)
    n = len(allCounts)
    wordlist = allCounts.keys()
    m = len(sens)
    print m
    ll = []
    for i in range(m):
        l = [0] * n
        for j in senCounts[i]:
            print j
            print wordlist.index(j)
            l[wordlist.index(j)] = senCounts[i][j]
        ll.append(l)
    print ll

def vectorSpace():
    # d1 = pd.read_csv("/Users/linxue/PycharmProjects/ml/resources/Youtube01-Psy.csv")
    d1 = pd.read_csv("Youtube01-Psy.csv")
    # for i in d1["CONTENT"]:

    lk = hasLink(d1["CONTENT"])
    #print lk
    ls = pd.Series(lk)
    #print ls
    d1 = pd.DataFrame({"CONTENT": d1["CONTENT"], "CLASS": d1["CLASS"], "LINK": ls})
    #print d1
    sens = []
    #d1list = pd.core.series.Series.tolist(d1["CONTENT"])
    for s in d1["CONTENT"]:
        s = ts.senTrim(s)
        a = s.split()
        for i in a:
            i = ts.wordTrim(i)
        sens.append(a)

    print type(sens)

    model = gensim.models.Word2Vec(sens, size=200, workers=4)
    model.save('w2v_model')


if __name__ == "__main__":
    #vectorSpace()
    new_model = gensim.models.Word2Vec.load('w2v_model')
    print type(new_model)


















