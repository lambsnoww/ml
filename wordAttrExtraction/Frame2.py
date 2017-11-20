#_*_coding:utf-8_*_

import json
import sys
from collections import Counter
from collections import defaultdict


# f = open("/Users/linxue/PycharmProjects/ml/resources/dataout.txt")
class Frame2(object):
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






if __name__ == "__main__":

    frame = Frame2(10, "/Users/linxue/PycharmProjects/ml/resources/dataout1.txt")
    print frame.framelist[1]
    print frame.framelist[10]
    print len(frame.framelist)











