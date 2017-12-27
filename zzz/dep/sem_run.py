#_*_coding:utf-8_*_
import numpy as np
import pandas as pd

if __name__ == '__main__':
    abb_phrase = ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X']
    f = open('sem_seq.txt', 'r')
    sems = f.readlines()
    f.close()

    f = open('labels.txt', 'r')
    labels = f.read().split()
    f.close()
    labels = [ord(i)-ord('a') for i in labels]
    vecs = []
    for i in sems:
        vec = [0] * len(abb_phrase) + 1
        for c in range(ord('a'), ord('z') + 1):
            x = chr(c)
            vec[ord(x) - ord('a')] = i.count(x)
        vecs.append(vec)







