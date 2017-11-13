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
        ls2 = []
        arr = f.readlines()
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
            l2 = sorted(l2, key=lambda x:x[1])
            ls.append(l)
            ls2.append(l2)

        fl = []
        for i in range(len(ls2)):
            t = ls2[i]
            count = 0
            fll = []
            for line in t:
                fll.append(line[0])
                count += 1
                if (count >= m):
                    break
            fl.append(fll)
        print fl


        self.framelist = fl
        self.vector = []
        self.countlist = [] # list of dicts
        self.allcount = {} # dict
        #print "self.framelist:"
        #print self.framelist
        ######add vector
        self.frame2vec()


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
        print "all frame keys:"
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
    print len(frame.vector[0])











