# -*- coding: utf-8 -*-
"""
Created on Saturday Dec 15 2018

@author: Yifei
"""

import numpy as np
import matplotlib.pylab as plt
from sklearn.utils import shuffle
import scipy.stats as stats
import math
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,explained_variance_score


def normalized(X):
    # rescaling features
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X)
    X_normalized = min_max_scaler.transform(X)
    return X_normalized

def confidence(prediction):
    # confidence intervals of samples
    # z_critical = stats.norm.ppf(q=0.95)  # Get the z-critical value*
    z = 1.96
    sd = prediction.std()
    mean = prediction.mean()
    margin_of_error = z * (sd / math.sqrt(len(prediction)))
    confidence_interval = (mean - margin_of_error,
                           mean + margin_of_error)
    return confidence_interval

def plot_confidence(Means,Intervals):
    fig, axs = plt.subplots(figsize=(18, 8))
    # ax = axs[0, 0]
    axs.errorbar(np.arange(1, len(Means) + 1), Means, yerr=[(top - bot) / 2 for top, bot in Intervals], fmt='o')
    axs.set_xlabel('Data Amount')
    axs.set_ylabel('Confidence Intervals')
    fig.savefig('Confidence Interval')

def plot(c,R,name):
    fig, ax =plt.subplots(figsize=(12,8))
    ax.plot(c, R)
    ax.set_xlabel('Data Amount')
    ax.set_ylabel(name)
    fig.savefig('result/%s.png'%name)


def main():
    #load the data
    data = np.loadtxt('data.csv', delimiter=',', skiprows=1, usecols=range(0, 15))
    headers = ['y', 'm', 'd', 'h', 'D', 'T', 'P', 'cv','NE','NW','SE','Iws', 'Is', 'Ir']
    X = data[:,0:14]
    y = data[:,14]
    m = len(X)

    # apply random forest to find the important features
    RF = RandomForestRegressor(n_estimators=10).fit(X, y)
    importance = RF.feature_importances_
    plt.bar(headers, importance)
    plt.savefig('Importance')
    #drop unimportance features
    X = np.delete(X, [7, 12], 1)

    # shuffle the data
    data_shuffle = shuffle(data, random_state=0)
    X = data_shuffle[:, 0:12]
    y = data_shuffle[:, 13]

    #normalized X
    X = normalized(X)

    # init
    Variance = []; R = []; Mean = []; Median = []
    Intervals = []
    Means = []

    #inscreasing data amount
    c = np.linspace(20, m, num=2000, endpoint=True, dtype=int)
    for i in c:
        X1 = X[0:i]
        y1 = y[0:i]

        # lasso regression
        #LR = linear_model.Lasso(alpha=0.1)
        # linear regression
        LR = linear_model.LinearRegression().fit(X1, y1)
        prediction = LR.predict(X)

        #adding mean and confidence intervals
        mean = prediction.mean()
        Means.append(mean)
        confidence_interval = confidence(prediction)
        Intervals.append(confidence_interval)

        #R^2 score
        r = LR.score(X,y)
        #print(r)
        R.append(r)

        #variance regression score
        v = explained_variance_score(y, prediction, multioutput='raw_values')
        Variance.append(v)

        #Mean absolute error
        mean = mean_absolute_error(y, prediction, multioutput='raw_values')
        Mean.append(mean)


    # plot the regression metrics
    # plot_final(c, Median, 'Median absolute error')
    plot(c, Variance, 'Variance regression score')
    plot(c, R, 'R-squared coefficient')
    plot(c, Mean, 'Mean absolute error')
    plot_confidence(Means,Intervals)


if __name__ == '__main__':
    main()
