#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:07:27 2023

@author: michael
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

plt.rcParams['figure.dpi'] = 150

fig,ax=plt.subplots(figsize=(4,0.75))
ax.scatter(np.around(np.array([0.02985, 0.05970, 0.14925, 0.29851, 0.44776,
                               0.53731, 0.59701, 0.74627, 0.89552, 0.98507])*33.5),
           np.array([0,0,0,0,0,0,0,0,0,0]),marker='.')
ax.set_title('Levels of Session 3 data used for calibration')
ax.set_xlabel('Gestures of each class')
ax.set_yticks([])
ax.tick_params(axis='x',which='minor',bottom=False,grid_linestyle='-',grid_alpha=0.25)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.grid(visible=True,axis='x',which='both')
ax.set_ylim(-0.2,0.2)
#plt.tight_layout();
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlim(-1,34)
plt.show()


fig,ax=plt.subplots(figsize=(4,0.75))
ax.scatter(np.around(np.array([0.02985, 0.05970, 0.14925, 0.29851, 0.44776,
                               0.53731, 0.59701, 0.74627, 0.89552, 0.98507])*134),
           np.array([0,0,0,0,0,0,0,0,0,0]),marker='.')
ax.set_title('Levels of Same-Session data')
ax.set_xlabel('Session 3 Gestures used for calibration')
ax.set_yticks([])
ax.tick_params(axis='x',which='minor',bottom=False,grid_linestyle='-',grid_alpha=0.25)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.grid(visible=True,axis='x',which='both')
ax.set_ylim(-0.2,0.2)
#plt.tight_layout();
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlim(-1,134)
plt.show()