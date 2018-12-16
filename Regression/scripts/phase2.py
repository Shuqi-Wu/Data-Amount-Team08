# -*- coding: utf-8 -*-
"""
Created on Saturday Dec 15 2018

@author: Yifei
"""

import numpy as np
import matplotlib.pylab as plt
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor



def plot(c,R,name):
    fig3, ax3 =plt.subplots(figsize=(12,8))
    ax3.plot(c, R)
    ax3.set_xlabel('Data Amount')
    ax3.set_ylabel(name)
    fig3.savefig('result/%s.png'%name)


def main():
    #load the data
    data = np.loadtxt('data.csv', delimiter=',', skiprows=1, usecols=range(0, 14))
    X = data[:,0:12]
    y = data[:,13]
    m = len(X)

    # shuffle the data
    data_shuffle = shuffle(data, random_state=0)
    X = data_shuffle[:, 0:12]
    y = data_shuffle[:, 13]

    #rescaling features
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X)
    X_normalized = min_max_scaler.transform(X)

    R = []; R_1 = []
    c = np.linspace(20, m, num=2000, endpoint=True, dtype=int)
    for i in c:
        X1 = X[0:i]
        y1 = y[0:i]
        #lasso regression
        LR = linear_model.Lasso(alpha=0.1)
        LR.fit(X_train, y_train)
        
        # random forest
        RF = RandomForestRegressor(n_estimators=10)
        RF.fit(X1, y1)

        r = LF.score(X, y)
        r_1 = RF.score(X, y)
        R.append(r)
        R_1.append(r_1)
        #print('score R^2: %.3f' % RF.score(X, y))

    plot(c,R,'R-squared coefficient of random forest')
    plot(c,R_1,'R-squared coefficient of lasso regression')
    print(X_train)
    print(y_train)




if __name__ == '__main__':
    main()
