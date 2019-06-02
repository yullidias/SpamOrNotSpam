#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import nltk
import numpy as np
import collections
import random
import re
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mlxtend.plotting import plot_decision_regions
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from enum import Enum, auto
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from unicodedata import normalize
from sklearn.feature_extraction.text import CountVectorizer
from itertools import product

nltk.download('punkt')
nltk.download('stopwords')

def readRootDir(file):
    try:
        with open(file, 'r') as f:
            rootDir = f.read()
            rootDir = rootDir[:-1] if rootDir.endswith("\n") else rootDir
            return rootDir if rootDir.endswith('/') else rootDir + '/'
        return None
    except Exception as e:
        print(e)
        return None

def plotData(X, y):
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()

    plt.scatter(X[pos,0], X[pos,1], s=80, c='r', marker='x', linewidths=1)
    plt.scatter(X[neg,0], X[neg,1], s=80, c='b', marker='o', linewidths=1)


FILE_WITH_PATH_BASE = 'pathBase.txt'
PATH_BASE = readRootDir(FILE_WITH_PATH_BASE)

train = loadmat(PATH_BASE + 'spamTrain.mat')

xtrain = train['X']
ytrain = train['y']

# print('X:', xtrain.shape)
# print('y:', ytrain.shape)

plotData(xtrain, ytrain)

# numpy.ravel(a) - A 1-D array, containing the elements of the input, is returned.

def classificadorSVM(kernel, xtrain, ytrain):
    print("Kernel: " + kernel)
    if kernel == 'linear':
        classifier = svm.SVC(kernel=kernel)
    else:
        classifier = svm.SVC(kernel=kernel, gamma='auto')

    classifier.fit(xtrain, ytrain.ravel())

    test = loadmat(PATH_BASE + 'spamTest.mat')
    xtest = test['Xtest']
    ytest = test['ytest'].ravel()

    prediction = classifier.predict(xtest)
    print(classification_report(ytest, prediction))

classificadorSVM('linear', xtrain, ytrain)

classificadorSVM('poly', xtrain, ytrain)

classificadorSVM('rbf', xtrain, ytrain)

classificadorSVM('sigmoid', xtrain, ytrain)
