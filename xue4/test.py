#_*_coding:utf-8_*_
from pycorenlp import StanfordCoreNLP
import Levenshtein
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import evaluate
from sklearn import metrics
import math
from collections import defaultdict
#多值比较只取最小值
def gaussian(dist, a=1, b=0, c=0.3):
    return a*math.e**(-(dist-b)**2/(2*c**2))
def knn_lin0(k, x_train, y_train, test):
    a = test.split(',')
    '''
    for i in a:
        if i.find('p') != -1 and i.find('h') > i.find('p'):
            return 1
        if i.find('p') != -1 and i.find('h') == -1:
            return 1
    return 0
    '''
    dis_list = []
    for index in range(len(x_train)):
        dis = 9999
        b = x_train[index].split(',')
        for i in a:
            for j in b:
                tmp = Levenshtein.distance(i, j)
                if dis > tmp:
                    dis = tmp
        dis_list.append((dis, index))
    nearest = sorted(dis_list, key=lambda x: x[0])
    #print nearest[:100]
    if k < len(nearest):
        ne = nearest[:k]
    else:
        ne = nearest

    summ = 0
    for i in ne:
        summ += y_train[i[1]] * gaussian(i[0])
    if summ * 2 >= len(ne):
        return 1
    else:
        return 0

def knn_lin(k, x_train, y_train, test):
    a = test.split(',')
    '''
    for i in a:
        if i.find('p') != -1 and i.find('h') > i.find('p'):
            return 1
        if i.find('p') != -1 and i.find('h') == -1:
            return 1
    return 0
    '''
    dis_list = []
    for index in range(len(x_train)):
        dis = 9999
        b = x_train[index].split(',')
        for i in a:
            for j in b:
                tmp = Levenshtein.distance(i, j)
                if dis > tmp:
                    dis = tmp
        dis_list.append((dis, index))
    nearest = sorted(dis_list, key=lambda x: x[0])
    # print nearest[:100]
    if k < len(nearest):
        ne = nearest[:k]
    else:
        ne = nearest

    summ = 0
    for i in ne:
        summ += y_train[i[1]]
    if summ * 2 >= len(ne):
        return 1
    else:
        return 0

#多值比较取平均值
def knn_lin2(k, x_train, y_train, test):
    a = test.split(',')
    dis_list = []
    for index in range(len(x_train)):
        dis = 0
        b = x_train[index].split(',')
        for i in a:
            for j in b:
                tmp = Levenshtein.distance(i, j)
                dis += tmp
        dis = float(dis)/(len(a) * len(b))
        dis_list.append((dis, index))
    nearest = sorted(dis_list, key=lambda x: x[0])
    if k < len(nearest):
        ne = nearest[:k]
    else:
        ne = nearest

    summ = 0
    for i in ne:
        summ += y_train[i[1]]
    if summ * 2 >= len(ne):
        return 1
    else:
        return 0

    # random i in a, average in b
def knn_lin3(k, x_train, y_train, test):
    a = test.split(',')
    dis_list = []
    for index in range(len(x_train)):
        dis = 0
        b = x_train[index].split(',')
        i = random.randint(0, len(a) - 1)
        for j in b:
            tmp = Levenshtein.distance(a[i], j)
            dis += tmp
        dis = float(dis) / (len(b))
        dis_list.append((dis, index))
    nearest = sorted(dis_list, key=lambda x: x[0])

    if k < len(nearest):
        ne = nearest[:k]
    else:
        ne = nearest

    summ = 0
    for i in ne:
        summ += y_train[i[1]]
    if summ * 2 >= len(ne):
        return 1
    else:
        return 0

#deprecated
def get_tags2(line):
    b = line.split('\n')
    print b
    d = []
    for i in b:
        print i
        if i != '':
            tmp = ''
            for j in range(len(i)):
                if (i[j]>= 'A' and i[j] <= 'Z'):
                    tmp += i[j]
                else:
                    if tmp != '':
                        d.append(tmp)
                        break
                    else:
                        break

def get_tags(line):
    a = line.split('\n')
    #print a
    lst = []
    for i in a:
        if i == '':
            continue
        beg = i.find('(')
        pos = i.find(' ', beg)
        #print pos
        if pos == -1:
            t = i[beg+1:]
        else:
            t = i[beg+1: pos]
        lst.append(t)
    for i in lst:
        if not i.isalpha():
            lst.remove(i)
    return lst

#by charactor a-z:
def get_sem_sequence(sens):
    rt = []
    for text in sens:
        print text

        while len(text) > 500:
            pos = text.rfind('.')
            pos = text.rfind('.',0,pos)
            if pos == -1:
                break
            text = text[0:pos + 1]
        if len(text) > 500:
            text = text[:500]


        output = nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit,pos,depparse,parse',
            'outputFormat': 'json'
        })

        tag = abb
        if 'sentences' in output:
            lst = []
            for i in range(len(output['sentences'])):
                a = output['sentences'][i]['parse']
                t = get_tags(a)

                for i in t:
                    if i in tag or i == 'ROOT':
                        lst.append(i)
                print lst
            rt.append(lst)
        else:
            rt.append([])
    return rt

def to_sequence(lines):
    tags = abb
    rt = []
    for line in lines:
        s = ''
        for tag in line:
            if tag in tags:
                index = tags.index(tag)
                stringa = chr(ord('a') + index)
                s += stringa
            elif tag == 'ROOT':
                s += ','
        rt.append(s[1:])
    return rt

def tostring(list):
    s = ''
    for i in list:
        s = s + ',' + str(i)
    return s[1:]

def hasLink(sens):
    #attLink = ""
    ls = []
    for sen in sens:
        l = 0
        if ("http://" in sen) or ("https://" in sen) or ("www." in sen) or (".com" in sen):
            l = 1
        elif sen.count('/') >= 3:
            l = 1
        # attLink = attLink + ',' + str(l)
        # new added
        if 'com' in sen.split():
            l = 1
        ls.append(l)
    return ls

def main():

    f = open('allsens.txt', 'r')
    sens = f.readlines()
    f.close()
    d = pd.read_csv('YouTube.csv')
    ls = hasLink(np.array(d['CONTENT']))
    d['LINK'] = pd.Series(ls)
    d['CONTENT'] = pd.Series(sens)
    print sens
    # COMMENT_ID,AUTHOR,DATE,CONTENT,CLASS
    d.drop('COMMENT_ID', axis=1)
    d.drop('AUTHOR', axis=1)
    d.drop('DATE', axis=1)

    df = d[d['LINK'] == 0]
    sens = df['CONTENT']

    y = np.array(df['CLASS'])

    print 'begin service'
    lines = get_sem_sequence(sens)
    tags = to_sequence(lines)

    f = open('sem_seq.txt', 'w')
    for i in tags:
        f.write(i + '\n')
    f.close()
def test():
    sens = ['''Just for test I have to say murdevcom.''','''Me shaking my sexy ass on my channel enjoy.''','''Now its 1884034783 views! Please comment the view count the next hour.''']
    lines = get_sem_sequence(sens)
    tags = to_sequence(lines)
    print tags

def run():
    d = pd.read_csv('YouTube.csv')
    ls = hasLink(np.array(d['CONTENT']))
    d['LINK'] = pd.Series(ls)
    # COMMENT_ID,AUTHOR,DATE,CONTENT,CLASS
    df = d[d['LINK'] == 0]
    y = np.array(df['CLASS'])
    print len(y)


    f = open('sem_seq.txt', 'r')
    x = f.readlines()
    print len(x)
    print x[0]
    f.close()

    seed = random.randint(1, 100000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=seed)

    y_pred = []
    for test in x_test:
        y_pred.append(knn_lin(15, x_train, y_train, test))
    y_pred2 = []
    for test in x_test:
        y_pred2.append(knn_lin2(15, x_train, y_train, test))
    y_pred3 = []
    for test in x_test:
        y_pred3.append(knn_lin3(15, x_train, y_train, test))
    y_pred0 = []
    for test in x_test:
        y_pred0.append(knn_lin0(15, x_train, y_train, test))
    evaluate.outcome(y_pred, y_test)
    evaluate.outcome(y_pred2, y_test)
    evaluate.outcome(y_pred3, y_test)
    evaluate.outcome(y_pred0, y_test)
    # print metrics.confusion_matrix(y_test, y_pred)
    # print metrics.confusion_matrix(y_test, y_pred2)


if __name__ == '__main__':

    abb_word = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
    abb_phrase = ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X']
    #abb = abb_word
    #abb.extend(abb_phrase)
    abb = ['CONJP', 'LST', 'NP', 'PP', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP']
    nlp = StanfordCoreNLP('http://localhost:9000')
    #main()
    #run()

    f = open('sem_seq.txt', 'r')
    lines = f.readlines()
    f.close()
    f = open('allsens.txt', 'r')
    sens = f.readlines()
    f.close()

    d = pd.read_csv('YouTube.csv')
    ls = hasLink(np.array(d['CONTENT']))
    d['LINK'] = pd.Series(ls)
    # COMMENT_ID,AUTHOR,DATE,CONTENT,CLASS
    df = d[d['LINK'] == 0]
    y = np.array(df['CLASS'])

    df.drop('COMMENT_ID', axis=1)
    df.drop('AUTHOR', axis=1)
    df.drop('DATE', axis=1)
    df.drop('CONTENT', axis=1)
    #df['SEQ'] = pd.Series(lines)
    #df['CONTENT'] = pd.Series(sens)
    #print df.head(100)

    sub = pd.read_csv('subjectivity.csv', header=None)
    emo = pd.read_csv('emotions.csv', header=None)
    e1 = []
    for i in df['CONTENT']:
        d = [0] * 5
        s = list(sub[0])
        for j in s:
            if j in i:
                tmp = sub[1][s.index(j)]
                tmp2 = sub[2][s.index(j)]
                a = 1
                b = 1
                if tmp == 'strongsubj':
                    a = 2
                else:
                    a = 1
                if tmp2 == 'positive':
                    b = 1
                else:
                    b = -1
                d[a * b + 2] += 1
        e1.append(d)

    e2 = []
    for i in np.array(df['CONTENT']):
        t = [0] * 7
        ind = ['anger', 'disgust', 'anger', 'joy', 'fear', 'sadness', 'surprise']

        for j in list(emo[0]):
            if j in i:
                tmp = emo[1][list(emo[0]).index(j)]
                #print ind.index(tmp)
                t[ind.index(tmp)] += 1
        e2.append(t)
    e = np.concatenate((e1,e2), axis=1)

    pd.DataFrame(e).to_csv('em.csv',header=None,index=None)























