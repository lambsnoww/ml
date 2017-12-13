#_*_coding:utf-8_*_

import re
import nlp
from textblob.classifiers import NaiveBayesClassifier
from pycorenlp import StanfordCoreNLP

def tostring(list):
    s = ''
    for i in list:
        s = s + ',' + str(i)
    return s[1:]

def tovector(fromfile, tofile):
    f1 = open(fromfile, 'r')
    lines = f1.readlines()
    f1.close()
    f = open(tofile, 'w')
    for text in lines:
        print text
        while len(text) > 300:
            text = text[:300]
            pos = text.rfind('.')
            if pos == -1:
                break
            text = text[:pos + 1]
        if len(text) > 300:
            text = text[:300]


            #text = text[:300]
        #output = nlp.semgrex(text, pattern='{tag: VBD}', filter=False)
        output = nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit,pos,depparse,parse',
            'outputFormat': 'json'
        })
        vector = [0] * len(abb)
        if 'sentences' in output:
            l = 0
            for i in range(len(output['sentences'])):
                a = output['sentences'][i]['parse']
                if a.find('VP') != -1 and a.find('NP') > a.find('VP'):
                    l += 1
                for j in range(len(abb)):
                    if abb[j] in a:
                        vector[j] += a.count(abb[j])
            f.write(tostring(vector) + ',' + str(l) + '\n')
        else:
            f.write(tostring([0] * len(abb)) + ',0' + '\n')
    f.close()


def abnormal():
    f1 = open('sens_final.txt', 'r')
    lines = f1.readlines()
    f1.close()
    f2 = open('abnormal.csv', 'w')
    for text in lines:
        f2.write(str(length_abnormal(text)) + ',' + str(punctuation_abnormal(text)) + '\n')
    f2.close()

def length_abnormal(sen):
    words = sen.split()
    if len(words) == 0:
        return 1
    a = float(len(sen)) / len(words)
    if a > 20 or a < 3:
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





def sem1(): #unused
    f1 = open('allsensclean.txt', 'r')
    lines = f1.readlines()
    f1.close()
    f = open('seminfo.csv', 'w')
    for text in lines:
        print text

        if len(text) > 300:
            text = text[:300]
        # output = nlp.semgrex(text, pattern='{tag: VBD}', filter=False)


        output = nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit,pos,depparse,parse',
            'outputFormat': 'json'
        })
        #vector = [0] * len(abb)
        l = 0
        if 'sentences' in output:
            for i in range(len(output['sentences'])):
                a = output['sentences'][i]['parse']
                if a.find('VP') != -1 and a.find('NP') > a.find('VP'):
                    l += 1

            f.write(str(l) + '\n')
        else:
            f.write(tostring([0] * len(abb)) + '\n')
    f.close()






if __name__ == '__main__':
    #abb = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'X']#add X
    abb_word = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
    abb_phrase = ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X']
    abb = abb_word
    abb.extend(abb_phrase)
    nlp = StanfordCoreNLP('http://localhost:9000')
    #tovector('allsensclean.txt')
    #tovector('sens_final.txt' 'sem.csv', abb)
    print abb
    tovector('sens_final.txt', 'sem_all.csv')
    abnormal()









