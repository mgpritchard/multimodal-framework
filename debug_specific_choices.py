# -*- coding: utf-8 -*-
"""
Created on Tue May 23 20:40:04 2023

@author: pritcham
"""


import testFusion as fuse
import handleML as ml

'''
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
'''

'''
space2=fuse.setup_search_space('decision',include_svm=True)

space2.update({'fusion_alg':'rf',
              'RFfuse':{
                'n_trees':25,
                'max_depth':5,
                },
              'emg':{'emg_model_type':'SVM_PlattScale',
                 'kernel':'rbf',
                 'svm_C':0.95933, 
                 'gamma':0.19836,
                 },
              
              'eeg':{'eeg_model_type':'LDA',
                     'LDA_solver':'lsqr',
                 'shrinkage':0.98376,
                 },
              })

emg_set=ml.pd.read_csv(space2['emg_set_path'],delimiter=',')
eeg_set=ml.pd.read_csv(space2['eeg_set_path'],delimiter=',')

space2.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'plot_confmats':True})
space2.update({'l1_sparsity':0.005})
space2.update({'l1_maxfeats':40})

results=fuse.function_fuse_withinppt(space2)
'''
space2=fuse.setup_search_space('featlevel',include_svm=True)

space2.update({
              'fusion_alg':'featlevel',
              'featfuse':
                    {'featfuse_model_type':'SVM_PlattScale', #keep this commented out
                     'kernel':'rbf',#'poly','linear']),
                     'svm_C':3.6806,
                     'gamma':0.0107,
                     },
              })

emg_set=ml.pd.read_csv(space2['emg_set_path'],delimiter=',')
eeg_set=ml.pd.read_csv(space2['eeg_set_path'],delimiter=',')

space2.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'plot_confmats':True})
space2.update({'l1_sparsity':0.005})
space2.update({'l1_maxfeats':40})

results=fuse.function_fuse_withinppt(space2)
print(results['fusion_mean_acc'])