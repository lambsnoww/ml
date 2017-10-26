#_*_coding:utf-8_*_

import json


def readFrame():
    f = open("/Users/linxue/PycharmProjects/ml/resources/dataout.txt")
    count = {}
    for line in f:
        #js = json.dumps(line, sort_keys=True, indent=4, separators=(',', ':'))
        #js = json.loads(line, encoding='utf-8')
        #js['frames'][0]['target']['name']
        #is the frame name of the sentence
        js = json.loads(line) #dumps-string; loads-dict
        str = js['frames'][0]['target']['name'].encode("gbk")
        print str
        count[str] += count[str]
        print type(str)
    print count




if __name__ == "__main__":
    readFrame()
