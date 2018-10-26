# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 22:27:15 2018

@author: Shuqi
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

def correlation(data):
    data=data.iloc[:,0:3]
    corr=data.corr()
    plt.matshow(corr)
    #data2=data.drop(['salary status'], axis=1)
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr,cmap=cmap, center=0, linewidths=.5)
        
    corr.to_csv('visualization/correlation.csv', sep=',')
    plt.savefig('visualization/correlation.png', dpi=500)
    return a

def main():
    
    #df = pd.read_csv('dataset2_formatted.csv',header=None,names=['ClusterID', 'Game mode', 'Game type'])
    df = pd.read_csv('dataset2_formatted.csv',header=None,names=['ClusterID', 'Game mode', 'Game type'])
   
    # plot the correlation
    a = correlation(df)


if __name__ == '__main__':
    main()
