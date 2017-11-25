#_*_coding:utf-8_*_

import re
import nlp
from textblob.classifiers import NaiveBayesClassifier
from pycorenlp import StanfordCoreNLP


if __name__ == '__main__':

    wordnlk = StanfordCoreNLP('http://localhost:9000')
    f = open('allsensclean.txt', 'r')
    lines = f.readlines()

    for text in lines[:10]:
        print text
        output = wordnlk.annotate(text, properties={
            'annotators': 'tokenize,ssplit,pos,depparse,parse',
            'outputFormat': 'json'
        })
        print(output['sentences'][0]['parse'])

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