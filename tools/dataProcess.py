#_*_coding:utf-8_*_

import numpy
import csv
from sklearn import svm
import tools.evaluate as ra
from sklearn.ensemble import RandomForestClassifier
import string
from contextlib import nested
import frameAnalyse as fa

def writeFile(filename, type):
    sen = []
    lab = []
    csv_reader = csv.reader(open(filename))

    if (type == "w"):
        f = open("/Users/linxue/PycharmProjects/ml/resources/data.txt", "w")
        f2 = open("/Users/linxue/PycharmProjects/ml/resources/labels.txt", "w")
        f3 = open("/Users/linxue/PycharmProjects/ml/resources/complete.txt", "w")
        f4 = open("/Users/linxue/PycharmProjects/ml/resources/attr.txt", "w")
        fp = open("/Users/linxue/PycharmProjects/ml/resources/fp.txt", "w")
        fn = open("/Users/linxue/PycharmProjects/ml/resources/fn.txt", "w")
    else:
        f = open("/Users/linxue/PycharmProjects/ml/resources/data.txt", "a")
        f2 = open("/Users/linxue/PycharmProjects/ml/resources/labels.txt", "a")
        f3 = open("/Users/linxue/PycharmProjects/ml/resources/complete.txt", "a")
        f4 = open("/Users/linxue/PycharmProjects/ml/resources/attr.txt", "a")
        fp = open("/Users/linxue/PycharmProjects/ml/resources/fp.txt", "a")
        fn = open("/Users/linxue/PycharmProjects/ml/resources/fn.txt", "a")

    flag = True
    for row in csv_reader:
        if flag:
            flag = False
            continue
        sen.append(row[3])
        lab.append(row[-1])
        #calculate attributes
        att = getAttr(row[3])
        f.write(row[3] + '\n')
        f2.write(row[-1] + '\n')
        f3.write(row[3] + ',' + row[-1] + ',' + str(att) + '\n')

        n = len(att)
        for i in att:
            n = n - 1
            f4.write(str(i))
            if n != 0:
                f4.write(',')
        f4.write('\n')
       # f4.write(str(att[0]) + ',' + str(att[1]) + ',' + str(att[2])  + ',' + str(att[3]) + '\n')

        if (row[-1] == '1'):
            fp.write(row[3] + '\n')
        else:
            fn.write(row[3] + '\n')

    f.close()
    f2.close()
    f3.close()
    f4.close()
    fp.close()
    fn.close()

def writeAttr():
    ff = open("/Users/linxue/PycharmProjects/ml/resources/attr_all.txt", 'w')

    f1 = open("/Users/linxue/PycharmProjects/ml/resources/attr.txt", 'r')
    f2 = open("/Users/linxue/PycharmProjects/ml/resources/dataout.txt", 'r')

    lines1 = f1.readlines()
    lines2 = f2.readlines()
    # print type(lines2)
    n = len(lines1)
    for i in range(n):
        s = fa.extractFrameAttribute(lines2[i])
        ff.write(lines1[i].strip() + ',' + s + '\n')

    ff.close()
    f1.close()
    f2.close()


def getAttr(s):

    # the num of words
    line = s.strip()
    words = line.split()
    word_counts = len(words)
    # if has a hyperlink
    link = 0
    if "http" in line:
        link = 1
    elif "www" in line:
        link = 1
    elif "html" in line:
        link = 1
    # elif "com" in line:
    #    link = 1

    # specific words: follow me, please subscribe, join, my channel, my videos, sub
    spwords = 0
    if "follow" in line:
        spwords += 1
    elif "subscribe" in line:
        spwords += 1
    elif "sub" in line:
        spwords += 1
    elif "join" in line:
        spwords += 1
    #elif "channel" in line:
    #    spwords += 1
    # elif "video" in line:
    #    spwords += 1
    spwords2 = 0
    if 'follow me' in line:
        spwords += 1
    elif 'please subscribe' in line:
        spwords += 1
    elif 'my channel' in line:
        spwords += 1
    elif 'my videos' in line:
        spwords += 1
    elif 'my new channel' in line:
        spwords += 1
    elif 'check out' in line:
        spwords += 1
    # uppercase
    pun = string.punctuation
    upcnt = 0
    for word in words:
        w = word.translate(None, pun)
        if word.isupper() and word != 'I':
            upcnt += 1
    # punctuation_____________________
    # print attr_line

    ret = numpy.array([word_counts, link, spwords, upcnt])
    # ret = numpy.array([link, spwords, upcnt])
    return ret

    # return attr

if __name__ == "__main__":
    # write to attr.txt the basic attributes of the data(without FrameNet)

    writeFile("/Users/linxue/PycharmProjects/ml/resources/Youtube01-Psy.csv", "w")
    writeFile("/Users/linxue/PycharmProjects/ml/resources/Youtube02-KatyPerry.csv", "a")
    writeFile("/Users/linxue/PycharmProjects/ml/resources/Youtube03-LMFAO.csv", "a")
    writeFile("/Users/linxue/PycharmProjects/ml/resources/Youtube04-Eminem.csv", "a")
    writeFile("/Users/linxue/PycharmProjects/ml/resources/Youtube05-Shakira.csv", "a")

    # write to attr_all.txt the additional attributes of the FrameNet
    writeAttr()

    print "data files have been written"


