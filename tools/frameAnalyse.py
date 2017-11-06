#_*_coding:utf-8_*_

import json
import sys

def frame2vec(filename):
    f = open(filename)
    for line in f:
        js = json.loads(line)
        n = len(js['frames'])
        ls = []
        l = []
        for i in range(n):
            str = js['frames'][i]['target']['name'].encode("gbk")
            l.extend(str)
        ls.append(l)

    return ls



def readFrame(filename):
    #f = open("/Users/linxue/PycharmProjects/ml/resources/fpout.txt")
    f = open(filename)
    count = {}
    for line in f:
        #js = json.dumps(line, sort_keys=True, indent=4, separators=(',', ':'))
        #js = json.loads(line, encoding='utf-8')
        #js['frames'][0]['target']['name']
        #is the frame name of the sentence
        js = json.loads(line) #dumps-string; loads-dict

        #print js['frames'][0]
        if js['frames'] != []:
            str = js['frames'][0]['target']['name'].encode("gbk")
        else:
            continue

        print str
        if str in count:
            count[str] += 1
        else:
            count[str] = 1
        print type(str)
    print count
    return count

def extractFrameAttribute(line):
    js = json.loads(line) #dumps-string; loads-dict
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    s5 = 0
    s6 = 0
    s7 = 0
    s8 = 0
    s9 = 0
    s10 = 0
    for i in range(min(10, len(js['frames']))):
        if js['frames'] != []:
            if js['frames'][i]['target']['name'].encode("gbk") == "People":
                s1 += 1
            elif js['frames'][i]['target']['name'].encode("gbk") == "Natural_features":
                s2 += 1
            elif js['frames'][i]['target']['name'].encode("gbk") == "Inspecting":
                s3 += 1
            elif js['frames'][i]['target']['name'].encode("gbk") == "Age":
                s4 += 1
            elif js['frames'][i]['target']['name'].encode("gbk") == "Experiencer_focus":
                s5 += 1
            elif js['frames'][i]['target']['name'].encode("gbk") == "Text":
                s6 += 1
            elif js['frames'][i]['target']['name'].encode("gbk") == "Desirability":
                s7 += 1
            elif js['frames'][i]['target']['name'].encode("gbk") == "Cardinal_numbers":
                s8 += 1
            elif js['frames'][i]['target']['name'].encode("gbk") == "Categorization":
                s9 += 1
            elif js['frames'][i]['target']['name'].encode("gbk") == "Locative_relation":
                s10 += 1
    return str(s1) + ',' + str(s2) + ',' + str(s3) + ',' + str(s4) + ',' + str(s5) + ',' + str(s6) + ',' + str(s7) + ',' + str(s8) + ',' + str(s9) + ',' + str(s10)

def calculateFrame():
    dp = {}
    dn = {}
    d = {}
    fp = open("/Users/linxue/PycharmProjects/ml/resources/fpout1.txt")
    fn = open("/Users/linxue/PycharmProjects/ml/resources/fnout1.txt")
    calculatelines(dp, fp.readlines(), 3)
    calculatelines(dn, fn.readlines(), 3)
    fp.close()
    fn.close()
    fp = open("/Users/linxue/PycharmProjects/ml/resources/fpout1.txt")
    fn = open("/Users/linxue/PycharmProjects/ml/resources/fnout1.txt")
    calculatelines(d, fn.readlines(), 3)
    calculatelines(d, fn.readlines(), 3)
    fp.close()
    fn.close()
    print "ddddddddddddddddddddddddddddddddddddddd\n"
    draw(d)
    print "ppppppppppppppppppppppppppppppppppppppp\n"
    draw(dp)
    print "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn\n"
    draw(dn)
    print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"


def draw(d):
    items = d.items()
    items = sorted(items, key=lambda x: x[1], reverse=True)
    #print items
    for i in items:
        print i[0] + ": "
        for j in range(i[1]):
            sys.stdout.write("#")
        print ""

    return 0


# called by calculateFrame
def calculatelines(d, lines, n):
    for line in lines:
        js = json.loads(line)
        for i in range(min(n, len(js['frames']))):
            if js['frames'] != []:
                framename =  js['frames'][i]['target']['name'].encode("gbk")
                if framename in d:
                    d[framename] += 1
                else:
                    d[framename] = 1

def getImportantFrames(filename, num):
    f = open(filename)
    count = {}
    ratio = {}
    for line in f:
        js = json.loads(line)
        n = min(num, len(js['frames']))
        if len(js['frames']) > 0:
            local_count = {}
            for i in range(min(num, len(js['frames']))):
                str = js['frames'][i]['target']['name'].encode("gbk")
                if str in count:
                    count[str] += 1
                else:
                    count[str] = 1

                if str in local_count:
                    local_count[str] += 1
                else:
                    local_count[str] = 1
            for i in range(min(num, len(js['frames']))):
                str = js['frames'][i]['target']['name'].encode("gbk")
                if str in ratio:
                    ratio[str] += local_count[str] / float(n)
                else:
                    ratio[str] = local_count[str] / float(n)
    #print count, ratio
    return count, ratio




if __name__ == "__main__":
    # readFrame("/Users/linxue/PycharmProjects/ml/resources/fpout.txt")
    #calculateFrame()
    fp = "/Users/linxue/PycharmProjects/ml/resources/fpout1.txt"
    cntp, rp = getImportantFrames(fp, 10)
    fn = "/Users/linxue/PycharmProjects/ml/resources/fnout1.txt"
    cntn, rn = getImportantFrames(fn, 10)

    diff = {}
    for key in rp:
        diff[key] = rp[key]
    for key in rn:
        if key in diff:
            diff[key] -= rn[key]
        else:
            diff[key] = rn[key] * (-1)
    items = diff.items()
    items = sorted(items, key=lambda x: x[1], reverse=True)
    print items





