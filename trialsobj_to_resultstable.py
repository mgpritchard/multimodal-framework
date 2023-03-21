#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:00:37 2023

@author: michael
"""

import testFusion as testfus
import testEEGClassify as testeeg
import pickle as pickle

def id_model_type(params):
    if params['eeg.LDA_solver']:
        return 'LDA'
    elif params['eeg.knn.k']:
        return 'knn'
    elif params['eeg.qda.regularisation']:
        return 'qda'
    elif params['eeg_ntrees']:
        return 'RF'

space=testeeg.setup_search_space()


filename='/home/michael/Documents/Aston/MultimodalFW/repo/multimodal-framework/lit_data_expts/waygal/results/EEGOnly/LOO/32ch_15pctfeats_trials_obj.p'
respath='/home/michael/Documents/Aston/MultimodalFW/repo/multimodal-framework/lit_data_expts/waygal/results/EEGOnly/LOO/32ch_15pctfeats_results.csv'

trials_obj=pickle.load(open(filename,'rb'))
results=trials_obj.trials.results
params=[trial['misc']['vals'] for trial in trials_obj.trials]

#results=[{k: v for k, v in d.items() if k != 'mykey1'} for d in results]
#https://stackoverflow.com/questions/13254241/removing-key-values-pairs-from-a-list-of-dictionaries
from operator import itemgetter
tuple_keys = ('median_acc','max_acc','max_acc_index')
get_keys = itemgetter(*tuple_keys)
results = [dict(zip(tuple_keys,get_keys(d))) for d in results]

for paramset in params:
    paramset.update({'eeg model':id_model_type(paramset)})
    
df=testfus.pd.concat([testfus.pd.DataFrame(params),testfus.pd.DataFrame(results)],axis=1)

df.to_csv(respath,index=False)