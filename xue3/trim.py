#_*_coding:utf-8_*_
from collections import defaultdict
import re

def trim_sens(infile, tofile): # 去乱符号并统一用句点代替，单词缩写改为全拼
    b = [',', '.', ':', ';', '?', '!']
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\n', '<', '>', '~', '_']

    f = open(infile, 'r')
    sens = f.readlines()
    #删除emoji
    sens = clean_emoji(sens)




    #把多个标点换成一个标点 and 句尾没有符号的给加上句点
    pattern = r'[,.:;?!~*]{2,}'
    re.compile(pattern)
    allsens = []
    for sen in sens:
        found = re.findall(pattern, sen)
        for i in found:
            tmp = i[0] + ' '
            sen = sen.replace(i, tmp)
        sen = sen.strip()
        if len(sen) >= 1 and (not sen[-1].isalpha()):
            sen = sen + '.'
        allsens.append(sen)
    sens = allsens


    abbd = defaultdict()
    abbd['ur'] = 'you are'
    abbd['u'] = 'you'
    abbd['im'] = 'i am'
    abbd['thx'] = 'thanks'
    abbd['plz'] = 'please'
    abbd['sub'] = 'subscribe'
    abbd['dont'] = 'don\'t'
    abbd['yr'] = 'year'

    fw = open(tofile, 'w')
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\n', '<', '>', '~', '_']
    a = ['@', '#', '$', '%', '^', '&', '*', '~']
    b = [',', '.', ':', ';', '?', '!']
    for sen in sens:
        ll = ''
        for word in sen.split():
            if word == '':
                continue
            tmp = clean(word.lower())
            if tmp == '':
                continue
            symbol = word[-1]
            if symbol in a:
                symbol = '.'
            if tmp in abbd:
                tmp = abbd[tmp]
            #if symbol.isalpha():
            #    symbol = ''
            if symbol not in b:
                symbol = ''
            ll = ll + tmp + symbol + ' '
        ll = ll.strip()
        if ll != '' and ll[-1].isalpha():
            ll = ll + '.'
        fw.write(ll + '\n')
    fw.close


def clean(word):
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\n', '<', '>', '~', '_']
    add = ['-', "'"]
    for i in english_punctuations:
        if i in word:
            word = word.replace(i, '')
    return word


def clean_emoji(sens):
    f = open('e.txt', 'r')
    es = f.read().splitlines()
    print es
    f.close()
    l = []
    sens_new = []
    for sen in sens:
        sen_new = ''
        c = 0
        words = sen.split()
        for e in es:
            if e in words:
                c += 1
                if words.index(e) > 0:
                    words[words.index(e) - 1] += '.'
                words.remove(e)
        for word in words:
            sen_new += word + ' '
        sen_new = sen_new.strip()
        sens_new.append(sen_new)
        l.append(c)

    return sens_new

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

def toUpper():
    f = open('sens_final.txt', 'r')
    sens = f.read().split()
    f.close()
    pun = ['.', '!', '?']
    words = ['i\'m', 'i am']
    out = []
    for sen in sens:
        new_sen = ''
        if len(sen) > 0:
            #sen.replace(0,str.toUpper(sen[0]))
            a = sen.split('.!?')
            for i in a:
                if len(i) > 0 and str.isalnum(i[0]):
                    if len(i) > 1:
                        i = i[0].upper() + i[1:]
                    else:
                        i = i[0].upper()
            b = ''.join(a)






if __name__ == '__main__':
    trim_sens('allsensclean.txt', 'sens_final.txt')