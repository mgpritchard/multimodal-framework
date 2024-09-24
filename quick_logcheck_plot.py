# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:21:38 2023

@author: pritcham
"""


from hyperopt import hp
import hyperopt.pyll.stochastic
import matplotlib.pyplot as plt
import numpy as np

logspace = hp.loguniform('xlog',1e-3,1e3)
loglogspace = hp.loguniform('xloglog',np.log(1e-3),np.log(1e3))

logsample=[]
loglogsample=[]

for i in range(1000):
    logsample.append(hyperopt.pyll.stochastic.sample(logspace))
    loglogsample.append(hyperopt.pyll.stochastic.sample(loglogspace))

#plt.plot(range(len(logsample)),np.sort(logsample),'r',label='log')
plt.plot(range(len(logsample)),np.sort(loglogsample),'g',label='loglog')
plt.yscale('log')
plt.legend()
plt.show()