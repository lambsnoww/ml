#_*_coding:utf-8_*_

import pandas as pd
import numpy as np
import tools
from sklearn.neural_network import MLPClassifier


d = pd.read_csv('Youtube.csv')

link = tools.hasLink(d['CONTENT'])
print len(d[d['CLASS']==1])
d['LINK'] = pd.Series(link)
'''
f = open('allsens.txt', 'r')
sens = f.readlines()
f.close()

d['SENS'] = pd.Series(sens)
d.drop('CONTENT', axis=1)
d.drop('COMMENT_ID', axis=1)
d.drop('AUTHOR', axis=1)
d.drop('DATE', axis=1)
df = d[d['LINK'] == 0]
f = open('sens_xlink.txt', 'w')
for i in df['SENS']:
    f.write(i)
f.close()
'''
