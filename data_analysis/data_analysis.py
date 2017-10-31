#_*_coding:utf-8_*_

import pandas as pd
import csv

#"/Users/linxue/PycharmProjects/ml/resources/all_data.txt"
#350+350+438+448(451)+370

# deprecated
def dataWrite():
    d1 = pd.read_csv("/Users/linxue/PycharmProjects/ml/resources/Youtube01-Psy.csv")
    d2 = pd.read_csv("/Users/linxue/PycharmProjects/ml/resources/Youtube02-KatyPerry.csv")
    d3 = pd.read_csv("/Users/linxue/PycharmProjects/ml/resources/Youtube03-LMFAO.csv")
    d4 = pd.read_csv("/Users/linxue/PycharmProjects/ml/resources/Youtube04-Eminem02.csv")
    d5 = pd.read_csv("/Users/linxue/PycharmProjects/ml/resources/Youtube05-Shakira.csv")

    d1.to_csv('all_data.csv', index = False, header = True)
    d2.to_csv('all_data.csv', index = False, header = False, mode = 'a+')
    d3.to_csv('all_data.csv', index = False, header = False, mode = 'a+')
    d4.to_csv('all_data.csv', index = False, header = False, mode = 'a+')
    d5.to_csv('all_data.csv', index = False, header = False, mode = 'a+')


def writeAttr():
    data = pd.read_csv("complete.csv")
    print len(data)







    











if __name__ == "__main__":
    # dataWrite()
    writeAttr()

