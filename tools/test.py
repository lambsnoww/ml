#_*_coding:utf-8_*_

from scipy.stats import ttest_ind
import pandas as pd
import numpy as np

if __name__ == "__main__":

    datap = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/fpout.txt"), delimiter=",", skiprows=0)
    datan = np.loadtxt(open("/Users/linxue/PycharmProjects/ml/resources/fnout.txt"), delimiter=",", skiprows=0)


