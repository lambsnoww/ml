#_*_coding:utf-8_*_
import string
import numpy

def getAttr(s):


    # the num of words
    line = s.strip()
    words = line.split()
    word_counts = len(words)


    # if has a hyperlink
    link = 0
    if "http" in line:
        link = 1
    elif "www" in line:
        link = 1
    elif "html" in line:
        link = 1
    # elif "com" in line:
    #    link = 1

    # specific words: follow me, please subscribe, join, my channel, my videos, sub
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
    # elif "video" in line:
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
    # uppercase
    pun = string.punctuation
    upcnt = 0
    for word in words:
        w = word.translate(None, pun)
        if word.isupper() and word != 'I':
            upcnt += 1
    # punctuation_____________________
    # print attr_line

    ret = numpy.array([word_counts, link, spwords, upcnt])
    # ret = numpy.array([link, spwords, upcnt])
    return ret

    # return attr

def get_word_features():
    f = open('sens_final.txt', 'r')
    a = f.readlines()
    ll = []
    for i in a:
        l = getAttr(i)
        ll.append(l)
    return numpy.array(ll)

if __name__ == '__main__':
    print get_word_features()