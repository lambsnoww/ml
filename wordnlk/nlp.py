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

def tovector():
    f1 = open('allsensclean.txt', 'r')
    lines = f1.readlines()
    f1.close()
    f = open('sem2.csv', 'w')
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


    '''    
        if len(output['sentences']) >= 1:
            a = output['sentences'][0]['parse']
            for i in range(len(abb)):
                if abb[i] in a:
                    vector[i] = a.count(abb[i])
            f.write(tostring(vector) + '\n')
        else:
            f.write(tostring([0] * len(abb)) + '\n')
    f.close()
    '''

    #    print(output['sentences'][0]['parse'])
    #print output

    #print(output['sentences'][0]['parse'])

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
    abb = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'X']#add X

 #   abb = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB',
           #clause level
 #           'S', 'SBAR', 'SBARQ', 'SINV', 'SQ',
           #phase level
 #          'ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X',
           #grammatical role
#           'DTV', 'LGS', 'PRD', 'PUT', 'SBJ', 'TPC']
    nlp = StanfordCoreNLP('http://localhost:9000')
    tovector()
    #sem1()









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