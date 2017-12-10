#_*_coding:utf-8_*_

import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
import tools.evaluate as ev
from sklearn.model_selection import train_test_split
import random


if __name__ == "__main__":
    sem = pd.read_csv('sem2.csv', header=None)
    d = pd.read_csv('Youtube.csv')
    x = sem.values
    y = np.array(d['CLASS'])

    seed = random.randint(1,1000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)


    A, P, R, F = ev.outcome(y_pred, y_test)