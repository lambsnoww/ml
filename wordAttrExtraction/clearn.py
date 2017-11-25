#_*_coding:utf-8_*_

import HTMLParser

def clean():
    f = open('allsens.txt', 'r')
    fout = open('allsensclean.txt', 'w')
    l = f.readlines()
    html_parser = HTMLParser.HTMLParser()
    for line in l:
        s = ''
        for i in range(len(line)):
            if (ord(line[i]) < 128 and ord(line[i]) > 0):
                s = s + line[i]
        s = html_parser.unescape(s)
        fout.write(s)


if __name__ == '__main__':
    clean()


