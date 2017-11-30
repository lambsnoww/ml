#_*_coding:utf-8_*_

import string
from string import *
from collections import Counter


def wordTrim(word):
    sy = "~!@#$%^&*()_+-={}[]|\\:;,.?<>\xea\xb0\x95\xeb\x82\xa8\xec\x8a\xa4\xed\x83\x80\xec\x9d\xbc\xc2\xb7\xef\xb4\xa9\xef\xb3\xc5\xab\xa1\xad\xae\xa3\xaf\xad\xad\xa5\xe2\x9c\xf0\x9f\x98\x97'"
    str = ""
    for i in range(len(word)):
        if word[i] in sy:
            continue
        else:
            str = str + word[i]
    str = str.lower()
    return str

def senTrim(sen):
    #words = sen.split("\|/|%|*|+~!@#$%^&*()_+-={}[]", ' ')
    sen = sen.replace('\xef\xbb\xbf','')#用replace删掉'\xef\xbb\xbf'
    words = sen.split(' ')
    str = ""
    for word in words:
        str = str + wordTrim(word) + " "
    str = str.strip()
    return str

def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
        lexicon.update([word for word in doc.split()])
    return lexicon

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

if __name__ == "__main__":
    '''
    f = open("/Users/linxue/PycharmProjects/ml/resources/data.txt", "r")
    list_of_all_file =
    vocabulary = build_lexicon(list_of_all_file)
    print 'the vector of two file is [' + ', '.join(list(vocabulary)) + ']'
    '''
    pass