#_*_coding:utf-8_*_

import json
import sys
from collections import Counter
from collections import defaultdict
import pandas as pd
import tools.wordProcess as tw

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
        self.stopwords = [] # stop words

        # get the sentences and words
        f = open("ENstopwords.txt", 'r')
        sw = []
        for i in f.readlines():
            sw.append(i.strip())
        self.stopwords = sw

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



        allkeys = allcount.keys()
        print "allkeys:"
        print allkeys


        vec = []
        for l in self.countlist:
            num = [0] * len(allkeys)
            for i in l.keys():
                num[allkeys.index(i)] += 1
            for i in l.keys():
                num[allkeys.index(i)] /= float(len(l))
            vec.append(num)

        self.vector = vec



if __name__ == "__main__":
    d1 = pd.read_csv("Youtube01-Psy.csv")
    lk = tw.hasLink(d1["CONTENT"])
    ls = pd.Series(lk)
    d1 = pd.DataFrame({"CONTENT": d1["CONTENT"], "CLASS": d1["CLASS"], "LINK": ls})
    word = Word(10, pd.Series.tolist(d1["CONTENT"]))
    print "sentences:"
    print word.sentences
    print "wordlist:"
    print word.wordlist












