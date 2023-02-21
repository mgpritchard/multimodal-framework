#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 12:34:04 2022

@author: michael
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_acc_comparison(pptlabels,emgAccs,eegAccs,fusAccs):
    x=np.arange(len(labels))
    width=0.36
    
    fig,ax=plt.subplots()
    rects1 = ax.bar(x - 2*width/3, emgAccs, 2*width/3, label='EMG')
    rects2 = ax.bar(x + 0, eegAccs, 2*width/3, label='EEG')
    rects3 = ax.bar(x + 2*width/3, fusAccs, 2*width/3, label='Fusion (Mean)')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy of different modes, leave-one-ppt-out')
    ax.set_xticks(x, pptlabels) #this version of mpl does not have labes in setxticks
    ax.set_ylim(0,1)
    ax.legend()
    
    #ax.bar_label(rects1, padding=3) #only in MPLB 3.4+
    #ax.bar_label(rects2, padding=3)
    #ax.bar_label(rects3, padding=3)
    
    fig.tight_layout()
    
    plt.show()
    
    
if __name__ == '__main__':
    
    labels=['1','2','4','7','8','9','11','13']
    emgAccs=[0.287,0.619,0.368,0.505,0.489,0.603,0.425,0.554]
    eegAccs=[0.163,0.163,0.176,0.159,0.161,0.153,0.166,0.152]
    fusAccs=[0.262,0.441,0.306,0.338,0.363,0.422,0.343,0.389]

    plot_acc_comparison(labels,emgAccs,eegAccs,fusAccs)