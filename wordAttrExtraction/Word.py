#_*_coding:utf-8_*_

import json
import sys
from collections import Counter
from collections import defaultdict
import pandas as pd
import tools.wordProcess as tw
import pylab


def ifStopword(word):
    f = open("ENstopwords.txt", 'r')
    for i in f.readlines():
        if word == i.strip():
            f.close()
            return True

    f.close()
    return False


def ifAbbrecation(word):
    f = open("ENabbrevation.txt", 'r')
    for i in f.readlines():
        a = i.split(',')
        if word == a[0]:
            f.close()
            return a[1]
    f.close()
    return False


# f = open("/Users/linxue/PycharmProjects/ml/resources/dataout.txt")
# Word - input is a list of sentences, and output is the trimed sentences & words
class Word(object):

    # m - max number of words for every sentences
    def __init__(self, m, contents):
        self.countlist = [] # list of dicts (word-count dict list of every sentences)
        self.allcount = defaultdict # dict (word-count dict of all sentences)
        self.wordlist = [] # list of list of words(trimed and without stopword)
        self.sentences = [] # list of trimed sentences
        self.vector = [] # word vector

        # get the sentences and words


        sens = []
        for s in contents:
            s = tw.senTrim(s)
            a = s.split()
            sen = ""
            for i in a:
                i = tw.wordTrim(i)
                sen = sen + ' ' + i
            sen = sen.strip()
            sens.append(sen)

        self.sentences = sens # trimed sentences

        ls = []
        for line in sens:
            words = line.split(' ')
            #n = min(len(words), m)
            n = len(words)
            l = []
            for word in words:
                l.append(word)
            ls.append(l)
        self.wordlist = ls # list of list of word in a sentences

    #def preprocess(self, m):
        allcount = defaultdict(int)
        countlist = []
        for l in self.wordlist: # self.wordlist
            count = defaultdict(int)
            for i in range(len(l)):
                allcount[l[i]] += 1
                count[l[i]] += 1
            countlist.append(count)
        self.countlist = countlist # list of dicts
        self.allcount = allcount # dict
        print "self.countlist"
        print self.countlist



        # sorted by number of value
        allkeys = []
        allkeys2 = sorted(allcount.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        for i in allkeys2:
            allkeys.append(i[0])

        #print "allkeys:"
        #print allkeys


        vec = []
        for l in self.countlist:
            num = [0] * len(allkeys)
            al = 0
            for i in l.keys():
                num[allkeys.index(i)] += l[i]
                al += l[i]
            for i in l.keys():
                num[allkeys.index(i)] /= float(al)
            vec.append(num)

        self.vector = vec





if __name__ == "__main__":
    d1 = pd.read_csv("Youtube01-Psy.csv")
    lk = tw.hasLink(d1["CONTENT"])
    ls = pd.Series(lk)
    d1 = pd.DataFrame({"CONTENT": d1["CONTENT"], "CLASS": d1["CLASS"], "LINK": ls})
    word = Word(10, pd.Series.tolist(d1["CONTENT"]))
    #print "sentences:"
    #print word.sentences
    #print "wordlist:"
    #print word.wordlist
    #print word.vector
    print len(word.vector[0])












