# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:00:14 2023

@author: pritcham
"""


import pickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_ppt_minmax_normalised(var,column,label=''):
    var_norm=(var-var.min())/(var.max()-var.min())
    var_norm.transpose().plot(title=(column+' minmax normalised'+label))
    var_norm['mean']=var_norm.mean(axis=1)
    print('mean of minmax normalised ',column,': ',str(var_norm['mean']),label)

def plot_ppt_rank(var,column,label=''):
    var_rank=var.rank()
    var_rank.transpose().plot(title=(column+' ranking'+label))
    fig,ax=plt.subplots()
    var_rank.transpose().boxplot(showmeans=True)#title='emg rank')

def plot_ppt_distro(table,column='emg_accs',label='',rank=False,minmaxnorm=False):
    var=table[column]
    var=pd.DataFrame.from_dict(dict(zip(var.index,var.values)))
    var.transpose().plot(title=(column+label))
    fig,ax=plt.subplots()
    var.transpose().boxplot(showmeans=True)
    if rank:
        plot_ppt_rank(var,column,label)
    if minmaxnorm:
        plot_ppt_minmax_normalised(var,column,label)


if __name__ == '__main__':
    resultformat='trials_obj'
    
    if resultformat=='trials_obj':
        resultpath='C:/Users/pritcham/Downloads/trials_obj_dec_LOO.p'
        
        trials=pickle.load(open(resultpath,'rb'))
        
        table=pd.DataFrame(trials.trials)
        table_readable=pd.concat(
        [pd.DataFrame(table['result'].tolist()),
         pd.DataFrame(pd.DataFrame(table['misc'].tolist())['vals'].values.tolist())],
        axis=1,join='outer')
        
    elif resultformat=='dataframe':
        rootpath=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Bespoke_100_compare"
        with open(os.path.join(rootpath,"emg_scores.pkl"),'rb') as f:
            table_readable=pickle.load(f)
    
    
    
    emg_accs=table_readable['emg_accs']
    emg_accs=pd.DataFrame.from_dict(dict(zip(emg_accs.index,emg_accs.values)))
    emg_accs.transpose().plot(title='emg acc')
    fig,ax=plt.subplots()
    emg_accs.transpose().boxplot(showmeans=True)
    
    
    emg_rank=emg_accs.rank()
    emg_rank.transpose().plot(title='emg ranking')
    fig,ax=plt.subplots()
    emg_rank.transpose().boxplot(showmeans=True)#title='emg rank')
    
    emg_norm=(emg_accs-emg_accs.min())/(emg_accs.max()-emg_accs.min())
    emg_norm.transpose().plot(title='emg acc minmax normalised')
    emg_norm['mean']=emg_norm.mean(axis=1)
    
    
    
    fig,ax=plt.subplots()
    eeg_accs=table_readable['eeg_accs']
    eeg_accs=pd.DataFrame.from_dict(dict(zip(eeg_accs.index,eeg_accs.values)))
    eeg_accs.transpose().plot(title='eeg acc')
    fig,ax=plt.subplots()
    eeg_accs.transpose().boxplot(showmeans=True)
    
    eeg_rank=eeg_accs.rank()
    eeg_rank.transpose().plot(title='eeg ranking')
    fig,ax=plt.subplots()
    eeg_rank.transpose().boxplot(showmeans=True)#title='eeg rank')
    
    eeg_norm=(eeg_accs-eeg_accs.min())/(eeg_accs.max()-eeg_accs.min())
    eeg_norm.transpose().plot(title='eeg acc minmax normalised')
    eeg_norm['mean']=eeg_norm.mean(axis=1)
    
    '''
    fig,ax=plt.subplots()
    fus_accs=table_readable['fusion_accs']
    fus_accs=pd.DataFrame.from_dict(dict(zip(fus_accs.index,fus_accs.values)))
    fus_accs.transpose().plot(title='fus acc')
    fig,ax=plt.subplots()
    fus_accs.transpose().boxplot(showmeans=True)
    
    fus_rank=fus_accs.rank()
    fus_rank.transpose().plot(title='fus ranking')
    fig,ax=plt.subplots()
    fus_rank.transpose().boxplot(showmeans=True)#title='eeg rank')
    
    fus_norm=(fus_accs-fus_accs.min())/(fus_accs.max()-fus_accs.min())
    fus_norm.transpose().plot(title='fus acc minmax normalised')
    fus_norm['mean']=fus_norm.mean(axis=1)
    '''