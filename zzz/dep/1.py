import pandas as pd
import numpy as np
import tools




df = pd.read_csv('Youtube.csv')

print df.columns

print len(df[df['CLASS'] == 1])
print len(df[df['CLASS'] == 0])


ls = tools.hasLink(df['CONTENT'])
df['LINK'] = pd.Series(ls)



print len(df[df['LINK'] == 1])
print len(df[df['LINK'] == 0])

nolink = df[df['LINK'] == 0]

print len(nolink[nolink['CLASS'] == 1])
print len(nolink[nolink['CLASS'] == 0])
