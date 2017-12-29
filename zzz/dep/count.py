#_*_coding:utf-8_*_

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

f = open('allsens.txt', 'r')
sens = f.readlines()
f.close()

link = hasLink(sens)
print len(link)
print sum(link)

