#_*_coding:utf-8_*_
from pycorenlp import StanfordCoreNLP
from sklearn.metrics import *
import nltk
from nltk.stem import WordNetLemmatizer
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
                    d = str.lower()

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

    return rt, vec, verblist, countlemma

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




