#_*_coding:utf-8_*_

import json
import sys
from collections import Counter
from collections import defaultdict


# f = open("/Users/linxue/PycharmProjects/ml/resources/dataout.txt")
class Frame(object):
    # m - max number of frames for every sentences
    def __init__(self, m, filename):
        f = open(filename)
        ls = []
        for line in f:
            js = json.loads(line)
            n = min(len(js['frames']), m)
            l = []
            for i in range(n):
                str = js['frames'][i]['target']['name'].encode("gbk")
                l.append(str)
            ls.append(l)

        self.framelist = ls
        self.vector = []
        self.countlist = [] # list of dicts
        self.allcount = {} # dict
        print "self.framelist:"
        print self.framelist


    def frame2vec(self):
        allcount = defaultdict(int)
        countlist = []
        for l in self.framelist:
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

    frame = Frame(10, "/Users/linxue/PycharmProjects/ml/resources/dataout1.txt")
    frame.frame2vec()
    print frame.vector











