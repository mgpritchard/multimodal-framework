# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 22:40:14 2023

@author: pritcham
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

fig,ax=plt.subplots(figsize=(4,0.75));
ax.scatter(np.around(np.array([0,0.00666,0.02,0.05263, 0.075, 0.1, 0.166, 0.33])*150,0),
           np.array([0,0,0,0, 0, 0, 0, 0]),marker='.');
ax.set_title('Levels of Other-Subject data included');
ax.set_xlabel('Gestures of each class from each other subject');
ax.set_yticks([]);
ax.tick_params(axis='x',which='minor',bottom=False,grid_linestyle='-',grid_alpha=0.25);
ax.xaxis.set_minor_locator(AutoMinorLocator(5));
ax.grid(visible=True,axis='x',which='both');
ax.set_ylim(-0.2,0.2);
#plt.tight_layout();
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlim(-0.5,25.5) #when only going up to 0.166 ie 25
ax.set_xlim(-1,51)
plt.show()


fig,ax=plt.subplots(figsize=(4,0.75));
ax.scatter(np.around(np.array([0.05,0.1,0.2575,0.505,0.7525,1.0])*100,0),
           np.array([0,0,0,0, 0, 0]),marker='.');
ax.set_title('Levels of subject data included');
ax.set_xlabel('Gestures of each class from subject');
ax.set_yticks([]);
ax.set_xticks(np.arange(0,110,10))
ax.tick_params(axis='x',which='minor',bottom=False,grid_linestyle='-',grid_alpha=0.25);#':',grid_alpha=0.75);
ax.xaxis.set_minor_locator(AutoMinorLocator(5));
ax.grid(visible=True,axis='x',which='both');
ax.set_ylim(-0.2,0.2);
#plt.tight_layout();
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlim(-1,101)
plt.show()


fig,ax=plt.subplots(figsize=(4,0.75));
ax.set_frame_on(False)
ax.scatter(np.array([0.05,0.1,0.2575,0.505,0.7525,1.0]),
           np.array([0,0,0,0, 0, 0]),marker='.');
#ax.set_title('Levels of subject data included');
ax.set_title('Scaling factors for downsampling subject');
ax.set_yticks([]);
ax.set_xticks(np.arange(0,1.1,0.1))
ax.tick_params(axis='x',which='minor',bottom=False,grid_linestyle='-',grid_alpha=0.25);
ax.xaxis.set_minor_locator(AutoMinorLocator(5));
ax.grid(visible=True,axis='x',which='both');
ax.set_ylim(-0.2,0.2);
ax.set_xlim(-0.01,1.01)
#plt.tight_layout();
plt.show()