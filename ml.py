#_*_coding:utf-8_*_

import numpy
import csv
from sklearn import svm
import tools.evaluate as ra
import tools.learn as ln
from sklearn.ensemble import RandomForestClassifier
import string

def extractAttr(filename, oname, type):
    sen = []
    lab = []
    csv_reader = csv.reader(open(filename))
    flag = True
    for row in csv_reader:
    #    print row[3]
    #    print row[4]
    #    print len(row)
        if flag:
            flag = False
            continue
        sen.append(row[3])
        lab.append(row[-1])
    n = len(sen)
    if (type == "w"):
        f = open("data.txt", "w")
        f2 = open("labels.txt", "w")
    else:
        f = open("data.txt", "a")
        f2 = open("labels.txt", "a")


    #print "n = " + str(n)
    for i in range(n):
        #print i
        f.write(sen[i] + '\n')
        f2.write(lab[i] + '\n')
    f.close()
    f2.close()
    attr = []
    c = -1
    for line in sen:
        c = c + 1
        #the num of words
        words = line.split()
        word_counts = len(words)

        #if has a hyperlink
        link = 0
        if "http" in line:
            link = 1
        elif "www" in line:
            link = 1
        elif "html" in line:
            link = 1
        elif "com" in line:
            link = 1
        #specific words: follow me, please subscribe, join, my channel, my videos, sub
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
        #elif "video" in line:
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
        #uppercase
        pun = string.punctuation
        upcnt = 0
        for word in words:
            w = word.translate(None, pun)
            if word.isupper() and word != 'I':
                upcnt += 1
        #punctuation_____________________
        #print attr_line
        #attr.append([word_counts, link, spwords, upcnt, lab[c]])
        attr.append([word_counts, link, spwords, upcnt])
        #print "label" + lab[c]
    if (type == 'w'):
        f = open(oname, 'w')
    else:
        f = open(oname, 'a') # a - append
    for line in attr:
        s = str(line[0]) + ',' + str(line[1]) + ',' + str(line[2]) + ',' + str(line[3]) + '\n'
        f.write(s)
    #return attr


def load_data(per):
    #extractAttr("2.csv", "data2.txt")
    train = numpy.loadtxt(open("attr_data.txt"), delimiter=",", skiprows=0)
    lab = numpy.loadtxt(open("labels.txt"), delimiter=",", skiprows=0)
    n = len(train)
    m = int(n * per)
    mm = n - m
    print type(train)
    test = train[m:n]
    train = train[0:m]
    lab_test = lab[m:n]
    lab = lab[0:m]

    print "*********"
    return train, lab, test, lab_test

def svmlearn(train, lab, test, lab_test, p):
    print len(train)
    print len(test)
    clf = svm.SVC()
    clf.fit(train, lab)
    pre = clf.predict(test)
    out = abs(pre - lab_test)
    n = len(test) + len(train)
    m = len(test)
    print float(m - sum(out))/m
    #print ra.outcome(lab_test, pre)
    f = open("out.txt" ,"a")
    f.write("SVM: " + str(p) + "\n")
    f.write(str(ra.outcome(pre, lab_test)))
    f.write("\n")
    f.close()

def randomForest(train, lab, test, lab_test, p):
    clf = RandomForestClassifier(n_estimators = 10)
    clf = clf.fit(train, lab)



if __name__ == "__main__":
    #extractAttr("1.csv", "data.txt")
    #extractAttr("2.csv", "data2.txt")
    #load_data
    #svmlearn(train, lab, test, lab_test)

    extractAttr("/Users/linxue/PycharmProjects/ml/resources/Youtube01-Psy.csv", "attr_data.txt", "w")
    extractAttr("/Users/linxue/PycharmProjects/ml/resources/Youtube02-KatyPerry.csv", "attr_data.txt", "a")
    extractAttr("/Users/linxue/PycharmProjects/ml/resources/Youtube03-LMFAO.csv", "attr_data.txt", "a")
    extractAttr("/Users/linxue/PycharmProjects/ml/resources/Youtube04-Eminem.csv", "attr_data.txt", "a")
    extractAttr("/Users/linxue/PycharmProjects/ml/resources/Youtube05-Shakira.csv", "attr_data.txt", "a")

    p = 0.5
    train, lab, test, lab_test = load_data(p)
    train2, lab2, test2, lab_test2h = ln.loadDataSequential(p)
    n = len(train)
    m = len(test)
    cnt = 0
    cnt2 = 0
    for i in range(n):
        if (train[i].all() != train2[i].all()):
            cnt = cnt + 1
            print train[i], train2[i]
            print "!!!!!!!!!!!!!!"
    for i in range(m):
        if (lab[i] != lab2[i]):
            print "!!!!!!!!!!!!!!"
            cnt2 = cnt + 1
    print cnt, cnt2
    svmlearn(train, lab, test, lab_test, p)

    print "##########"