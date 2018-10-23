# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 2018

@author: Yifei Fang
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


def correlation(data1):
    data2 = data1.drop(['pm2.5'], axis=1)
    m = ['y', 'm', 'd', 'h', 'D', 'T', 'P', 'Iws', 'Is', 'Ir']
    a = data2.columns
    cor = data2.corr()
    plt.matshow(cor)
    plt.xticks(range(len(a)), m)
    plt.yticks(range(len(a)), a)
    plt.colorbar()
    cor.to_csv('visualization/correlation.csv', sep=',')
    plt.savefig('visualization/correlation.png', dpi=500)
    return a

def main():
    #load the data
    data1 = pd.read_csv('data.csv',sep=",")
    data = np.loadtxt('data.csv', delimiter=',', skiprows=1, usecols=range(0, 11))
    data = data.transpose()
    X = data[0:10,:]
    y = data[10,:]

    # plot the correlation
    a = correlation(data1)

    #plot each scatters of data
    for i in range(len(a)):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(X[i], y, alpha=0.5, label=a[i])
        ax.set_xlabel(a[i])
        ax.set_ylabel('PM2.5')
        ax.set_title('Beijing PM2.5 Data Data Set')
        fig.savefig('visualization/scatter %s.png'%a[i])

     # plot each scatters of data
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(len(a)):
        ax.scatter(X[i], y, alpha=0.5, label=a[i])
        ax.set_xlabel('X')
        ax.set_ylabel('PM2.5')
        ax.set_title('Beijing PM2.5 Data Data Set')
        fig.savefig('visualization/scatter %s.png' % a[i])



if __name__ == '__main__':
    main()
