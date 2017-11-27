#_*_coding:utf-8_*_

from pycorenlp import StanfordCoreNLP
import re
import json
nlp = StanfordCoreNLP('http://localhost:9000')

text = (
    'Pusheen and Smitha walked along the beach. '
    'Pusheen wanted to surf, but fell off the surfboard.')
#text = 'why not check out my channel'
#text = 'Please check out my channel'
output = nlp.annotate(text, properties={
    'annotators': 'tokenize,ssplit,pos,depparse,parse',
    'outputFormat': 'json'
})
print type(output)
s = output['sentences'][0]['parse']
s2 = output['sentences'][1]['parse']
print (s)
print (s2)
print json.dumps(output)
#w = re.split('[() ]', s)
#w = [word for word in w if word != '' and word != '\n']
#print w

#找到第一个VP，如果前面没有NN，后面有NN

# print output
abb = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
       'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
       'RBS', 'RP', 'SYM', 'TO', 'UH',
       'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']








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
















'''
import corenlp
text = 'I have a nice day. The sky is so clear.'
with corenlp.CoreNOLCline(annotators='tokenize ssplit pos'.split()) as client:
    ann = client.annotate(text)
    sentence = ann.sentence[0]

    for token in sentence.token:
        print token.word, token.pos
'''
