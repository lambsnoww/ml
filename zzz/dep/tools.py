#_*_coding:utf-8_*_
from pycorenlp import StanfordCoreNLP
from sklearn.metrics import *
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string
import numpy as np
#import nltk
#nltk.download()

abb_word = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
            'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
            'WDT', 'WP', 'WP$', 'WRB']
abb_phrase = ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'QP', 'RRC', 'UCP',
              'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X']
tags = abb_phrase

abb = abb_word
abb.extend(abb_phrase)
def wordTrim(word):
    new_word = ''
    for i in word:
        if i.isalpha():
            new_word += i
    return new_word

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

def lcs(s1, s2):
    m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]  #生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax=0   #最长匹配的长度
    p=0  #最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                m[i+1][j+1]=m[i][j]+1
                if m[i+1][j+1]>mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return mmax

def get_sem_sequence_vector(sens):
    rt = [] # list of tag list
    vec = [] # vector of tags
    verblist = [] # first verb base list
    countlemma = [] # base verb count


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

        nlp = StanfordCoreNLP('http://localhost:9000')

        output = nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit,pos,depparse,parse',
            'outputFormat': 'json'
        })

        lemmatizer = WordNetLemmatizer()
        cnt = 0
        verb = 'zzZ'
        if 'sentences' in output:
            lst = []
            for i in range(len(output['sentences'])):
                a = output['sentences'][i]['parse']
                t = get_tags(a)

                for ii in t:
                    if ii in tags or ii == 'ROOT':
                        lst.append(ii)
                print lst
                # get word before lemma
                if 'VP' in lst:
                    b = a.split()
                    for iii in b:
                        if 'VP' in iii:
                            c = iii
                            break
                    d = c.split()[-1]
                    for iiii in range(len(d)):
                        if d[-iiii-1].isalpha():
                            break
                    d = d[:-iiii]
                    d = str.lower(str(d))

                    if lemmatizer.lemmatize(d) == d:
                        cnt += 1
                        verb = verb + ' ' + lemmatizer.lemmatize(d, 'v')
                        #verb.append(lemmatizer.lemmatize(d))

            rt.append(lst)
        else:
            rt.append([])
        verblist.append(verb)
        countlemma.append(cnt)
        # extract vector
        p = [0] * len(abb)
        if 'sentences' in output:
            for i in range(len(output['sentences'])):
                a = output['sentences'][i]['parse']
                for t in abb:
                    if t in a:
                        p[abb.index(t)] += a.count(t)
        vec.append(p)

    return rt, vec, verblist, countlemma, notwordcount

#提取每一行开头的tag
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
def to_sequence(lines):

    rt = []
    for line in lines:
        s = ''
        for tag in line:
            if tag in tags:
                index = tags.index(tag)
                stringa = chr(ord('a') + index)
                s += stringa
            elif tag == 'ROOT' or tag == 'CC':
                s += ','
        rt.append(s[1:])
    return rt

def evaluate(y_pred, y_true):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f = f1_score(y_true, y_pred)

    print 'accuracy, precision, recall, f-measure'
    print acc, pre, rec, f

def abnormal(sens):
    abnlist = []
    a = 0; b = 0
    for sen in sens:
        if length_abnormal(sen):
            a = 1
        if punctuation_abnormal(sen):
            b = 1
        abnlist.append([a, b])
    return abnlist

def length_abnormal(sen):
    words = sen.split()
    maxlen = 0
    for i in words:
        if len(i) > maxlen:
            maxlen = len(i)

    if len(words) == 0:
        return 1
    a = float(len(sen)) / len(words)
    if a > 20 or a < 3:
        return 1
    elif maxlen > 20 or maxlen < 3:
        return 1
    return 0

def punctuation_abnormal(sen):
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\n', '<', '>', '~', '_']
    c = 0
    for i in english_punctuations:
        c += sen.count(i)
    if c == 0 and len(sen.split()) > 20:
        return 1
    if float(len(sen.split())) / c > 20:
        return 1
    return 0

def get_word_feature(sens):
    f = []
    for s in sens:

        # the num of words
        line = s.strip()
        line = line.lower()
        words = line.split()
        word_counts = len(words)


        # if has a hyperlink
        #link = 0
        spwords = 0
        if "http" in line:
            spwords += 1
        elif "www" in line:
            spwords += 1
        elif "html" in line:
            spwords += 1
        # elif "com" in line:
        #    link = 1

        # specific words: follow me, please subscribe, join, my channel, my videos, sub
        if "follow" in line:
            spwords += 1
        elif "subscribe" in line:
            spwords += 1
        elif "sub" in line:
            spwords += 1
        elif "join" in line:
            spwords += 1
        elif "check" in line:
            spwords += 1
        elif "please" in line:
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
        elif 'my video' in line:
            spwords += 1
        elif 'my new channel' in line:
            spwords += 1
        elif 'check out' in line:
            spwords += 1
        elif 'take a look at' in line:
            spwords += 1
        elif 'like this comment' in line:
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

        #ret = np.array([word_counts, link, spwords, upcnt])
        ret = np.array([word_counts, spwords, upcnt])
        # ret = np.array([link, spwords, upcnt])
        f.append(ret)
    return f

    # return attr

def print_misclassified(y_pred, y_test, x_test_sens):
    for i in range(len(y_test)):
        if y_test[i] == 0 and y_pred[i] == 1:
            print '1, 0'
            print x_test_sens[i]
    for i in range(len(y_test)):
        if y_test[i] == 1 and y_pred[i] == 0:
            print '0, 1'
            print x_test_sens[i]

    print confusion_matrix(y_test, y_pred)
