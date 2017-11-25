#_*_coding:utf-8_*_

import re
#import nlp
from textblob.classifiers import NaiveBayesClassifier
from pycorenlp import StanfordCoreNLP

def tostring(list):
    s = ''
    for i in list:
        s = s + ',' + str(i)
    return s[1:]

if __name__ == '__main__':

    abb = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
           'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
           'RBS', 'RP', 'SYM', 'TO', 'UH',
           'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

    nlp = StanfordCoreNLP('http://localhost:9000')
    f1 = open('allsensclean.txt', 'r')
    lines = f1.readlines()
    f1.close()
    f = open('ann.csv', 'w')
    for text in lines:
        print text

        if len(text) > 300:
            text = text[:300]
        #output = nlp.semgrex(text, pattern='{tag: VBD}', filter=False)





        output = nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit,pos,depparse,parse',
            'outputFormat': 'json'
        })
        vector = [0] * len(abb)
        if len(output['sentences']) >= 1:
            a = output['sentences'][0]['parse']
            for i in range(len(abb)):
                if abb[i] in a:
                    vector[i] = a.count(abb[i])
            f.write(tostring(vector) + '\n')
        else:
            f.write(tostring([0] * len(abb)) + '\n')
    f.close()

    #    print(output['sentences'][0]['parse'])
    #print output

    #print(output['sentences'][0]['parse'])








    '''

    text = (
        'Pusheen and Smitha walked along the beach. Pusheen wanted to surf,'
        'but fell off the surfboard.')
    output = wordnlk.annotate(text, properties={
        'annotators': 'tokenize,ssplit,pos,depparse,parse',
        'outputFormat': 'json'
    })
    print(output['sentences'][0]['parse'])
    output = wordnlk.tokensregex(text, pattern='/Pusheen|Smitha/', filter=False)
    print(output)
    output = wordnlk.semgrex(text, pattern='{tag: VBD}', filter=False)
    print(output)
    
    '''