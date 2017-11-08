#_*_coding:utf-8_*_

import pandas as pd
import numpy as np
import tools.wordProcess as tw
from Word import *
from Frame import *

if __name__ == "__main__":
    d1 = pd.read_csv("Youtube01-Psy.csv")
    lk = tw.hasLink(d1["CONTENT"])
    ls = pd.Series(lk)
    d1 = pd.DataFrame({"CONTENT": d1["CONTENT"], "CLASS": d1["CLASS"], "LINK": ls})
    word = Word(10, pd.Series.tolist(d1["CONTENT"]))

    # write sentences to file1.txt for Semafor to process
    f = open('file1.txt', 'w')
    for sen in d1["CONTENT"]:
        f.write(sen + '\n')
    f.close()

    # frame
    frame = Frame(10, 'f1out.txt')

    print len(word.vector)
    print len(frame.vector)
    all = np.hstack((word.vector, frame.vector))
    a = all.tolist()
    w = pd.Series(word.vector)
    f = pd.Series(frame.vector)
    d1 = pd.DataFrame({"CONTENT": d1["CONTENT"], "WORD": w, "FRAME": f, "WF": a, "LINK": ls, "CLASS": d1["CLASS"]})
    dw = pd.DataFrame({"WORD": w, "CLASS": d1['CLASS']})
    df = pd.DataFrame({"FRAME": f, "CLASS": d1['CLASS']})
    da = pd.DataFrame({"WF": a, "CLASS": d1['CLASS']})






    # frame = Frame(10, pd.Series.tolist(d1["CONTENT"]))
