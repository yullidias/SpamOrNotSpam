#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE

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
    plt.show()


FILE_WITH_PATH_BASE = 'pathBase.txt'
PATH_BASE = readRootDir(FILE_WITH_PATH_BASE)

train = loadmat(PATH_BASE + 'spamTrain.mat')
xtrain = train['X']
ytrain = train['y']

X_embedded = TSNE(n_components=2).fit_transform(xtrain)
plotData(X_embedded,ytrain)

test = loadmat(PATH_BASE + 'spamTest.mat')
xtest = test['Xtest']
ytest = test['ytest']

X_embedded = TSNE(n_components=2).fit_transform(xtest)
plotData(X_embedded,ytest)


def classificadorSVM(kernel, xtrain, ytrain):
    print("Kernel: " + kernel)
    if kernel == 'linear':
        classifier = svm.SVC(kernel=kernel)
    else:
        classifier = svm.SVC(kernel=kernel, gamma='auto')

    classifier.fit(xtrain, ytrain.ravel())

    prediction = classifier.predict(xtest)
    print(confusion_matrix(ytest, prediction))  
    print(classification_report(ytest, prediction))

classificadorSVM('linear', xtrain, ytrain)
classificadorSVM('poly', xtrain, ytrain)
classificadorSVM('rbf', xtrain, ytrain)
classificadorSVM('sigmoid', xtrain, ytrain)





