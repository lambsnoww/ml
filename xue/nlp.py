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
        ls.append(l)

    return ls


def abnormal(sen):
    lensum = 0
    count = 0
    for word in sen.split():
        lensum += len(word)
        count += 1
    average = float(lensum) / count
    if average < 2:
        return True
    elif average > 15:
        return True
    else:
        return False



if __name__ == '__main__':
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\n', '<', '>', '~', '-', '_']



