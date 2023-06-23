# -*- coding: utf-8 -*-
"""
Created on Tue May 23 20:40:04 2023

@author: pritcham
"""


import testFusion as fuse
import handleML as ml

space=fuse.setup_search_space('decision')

space.update({'fusion_alg':'mean',
              'emg':{'emg_model_type':'kNN',
                 'knn_k':17},
              'eeg':{'eeg_model_type':'gaussNB',
                     'smoothing':0.8819134694424081},
              'scalingtype':'standardise'})

emg_set=ml.pd.read_csv(space['emg_set_path'],delimiter=',')
eeg_set=ml.pd.read_csv(space['eeg_set_path'],delimiter=',')

space.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'plot_confmats':True})

results=fuse.function_fuse_LOO(space)
