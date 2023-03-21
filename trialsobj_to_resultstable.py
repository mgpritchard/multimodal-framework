#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:00:37 2023

@author: michael
"""

import testFusion as testfus
import testEEGClassify as testeeg
import pickle as pickle
from operator import itemgetter


def id_EEG_model_type(params):
    if params['eeg.LDA_solver']:
        return 'LDA'
    elif params['eeg.knn.k']:
        return 'knn'
    elif params['eeg.qda.regularisation']:
        return 'qda'
    elif params['eeg_ntrees']:
        return 'RF'
    
def id_EMG_model_type(params):
    if params['emg.LDA_solver']:
        return 'LDA'
    elif params['emg.knn.k']:
        return 'knn'
    elif params['emg.qda.regularisation']:
        return 'qda'
    elif params['emg.RF.ntrees']:
        return 'RF'
    
    

space=testeeg.setup_search_space()


#filename='/home/michael/Documents/Aston/MultimodalFW/repo/multimodal-framework/lit_data_expts/waygal/results/EEGOnly/LOO/32ch_15pctfeats_trials_obj.p'
#respath='/home/michael/Documents/Aston/MultimodalFW/repo/multimodal-framework/lit_data_expts/waygal/results/EEGOnly/LOO/32ch_15pctfeats_results.csv'
filename='C:/users/pritcham/Documents/mm-framework/multimodal-framework/lit_data_expts/waygal/results/Fusion_32EEG/LOO/trials_obj.p'
respath='C:/users/pritcham/Documents/mm-framework/multimodal-framework/lit_data_expts/waygal/results/Fusion_32EEG/LOO/results.csv'
mode='fusion'

trials_obj=pickle.load(open(filename,'rb'))
if mode=='EEG':
    results=trials_obj.trials.results
elif mode=='fusion':
    results=trials_obj.results
params=[trial['misc']['vals'] for trial in trials_obj.trials]

#results=[{k: v for k, v in d.items() if k != 'mykey1'} for d in results]
#https://stackoverflow.com/questions/13254241/removing-key-values-pairs-from-a-list-of-dictionaries
if mode=='EEG':
    tuple_keys = ('median_acc','max_acc','max_acc_index')
elif mode=='fusion':
    tuple_keys=('fusion_mean_acc','fusion_median_acc','emg_mean_acc','eeg_mean_acc','elapsed_time')
get_keys = itemgetter(*tuple_keys)
results = [dict(zip(tuple_keys,get_keys(d))) for d in results]

for paramset in params:
    if 'eeg model' in paramset:
        paramset.update({'eeg model':id_EEG_model_type(paramset)})
    if 'emg model' in paramset:
        paramset.update({'emg model':id_EMG_model_type(paramset)})
    
df=testfus.pd.concat([testfus.pd.DataFrame(params),testfus.pd.DataFrame(results)],axis=1)

df.to_csv(respath,index=False)