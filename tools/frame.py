#_*_coding:utf-8_*_

import json


def readFrame():
    f = open("/Users/linxue/PycharmProjects/ml/resources/dataout.txt")
    cnt = 0
    for line in f:
        if cnt == 10:
            break
        #js = json.dumps(line, sort_keys=True, indent=4, separators=(',', ':'))
        #js = json.loads(line, encoding='utf-8')
        js = json.loads(line)
        str = js['frames'][0]['target']['name'].encode("gbk")
        print str
        print type(str)
        cnt += 1
        #js['frames'][0]['target']['name']




if __name__ == "__main__":
    readFrame()
