#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:36:23 2021

@author: pritcham

prototype script to plot probability distributions over time.
adapted from https://matplotlib.org/stable/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html
"""

import numpy as np
import matplotlib.pyplot as plt

classlabels=['close','open','neutral','grip','lateral','tripod']

results = [[0.1,0.1,0.1,0.1,0.6],
           [0.1,0.1,0.2,0.1,0.5],
           [0.1,0.1,0.4,0.1,0.3],
           [0.01,0.05,0.5,0.19,0.25],
           [0.001,0.049,0.6,0.15,0.2],
           [0.05,0.05,0.65,0.1,0.15],
           [0.05,0.05,0.79,0.01,0.1],
           [0.05,0.05,0.74,0.01,0.15],]

def plotdist(data,labels):
    data=np.stack(np.asarray(data).squeeze())
    data_cumul=data.cumsum(axis=1)
    labelcolours=plt.get_cmap('RdYlGn')(np.linspace(0.15,0.85,data.shape[1]))
    
    fig,ax=plt.subplots(figsize=(9.2,8))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0,np.sum(data,axis=1).max())
    
    instances=range(data.shape[0])
    
    for i, (colname, color) in enumerate(zip(labels,labelcolours)):
        widths=data[:,i]
        starts=data_cumul[:,i]-widths
        rects=ax.barh(instances,widths,left=starts,height=1.0,
                      label=colname,color=color)
        r,g,b,_=color
        text_color='white' if r*g*b<0.5 else 'darkgrey'
        #ax.bar_label(rects,label_type='center',color=text_color)
        
    ax.legend(ncol=len(classlabels),bbox_to_anchor=(0,1),
                  loc='lower left',fontsize='small')     
    return fig,ax
    
    
if __name__=='__main__':
    plotdist(results,classlabels)
    plt.show()