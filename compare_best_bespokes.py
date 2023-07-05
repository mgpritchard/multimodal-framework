# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 18:03:44 2023

@author: pritcham
"""

import testFusion as fuse
import handleML as ml
import params as params

num_trials=100

emg_set=ml.pd.read_csv(params.jeong_EMGfeats,delimiter=',')
eeg_set=ml.pd.read_csv(params.jeong_noCSP_WidebandFeats,delimiter=',')
emg_set,eeg_set=fuse.balance_set(emg_set,eeg_set)


space_just_emg=fuse.setup_search_space('just_emg',include_svm=True)
space_just_emg.update({'fusion_alg':'just_emg',
              'emg':{'emg_model_type':'SVM_PlattScale',
                 'kernel':'rbf',
                 #'svm_C':10.8608, 
                 'svm_C':10.86077990984692,
                 #'gamma':0.0121,
                 'gamma':0.01207463833393443,
                 },
              })
space_just_emg.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt'})
emg_scores=[]
for n in range(num_trials):
    results=fuse.function_fuse_withinppt(space_just_emg)
    emg_scores.append(results)
emg_scores=ml.pd.DataFrame(emg_scores)
print('Mean mean EMG accuracy: ',str(emg_scores['emg_mean_acc'].mean()))



space_just_eeg=fuse.setup_search_space('just_eeg',include_svm=True)
space_just_eeg.update({'fusion_alg':'just_eeg',
              'eeg':{'eeg_model_type':'LDA',
                 'LDA_solver':'lsqr',
                 #'shrinkage':0.2445, 
                 'shrinkage':0.24453863248230145,
                 },
              })
space_just_eeg.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt'})
eeg_scores=[]
for n in range(num_trials):
    results=fuse.function_fuse_withinppt(space_just_eeg)
    eeg_scores.append(results)
eeg_scores=ml.pd.DataFrame(eeg_scores)
print('Mean mean EEG accuracy: ',str(eeg_scores['eeg_mean_acc'].mean()))



space_decision=fuse.setup_search_space('decision',include_svm=True)
space_decision.update({'fusion_alg':'LDA',
                       'ldafuse':{'LDA_solver':'eigen',
                                  #'shrinkage':0.216787
                                  'shrinkage':0.21678705982755336,},
                       'eeg':{'eeg_model_type':'kNN',
                              'knn_k':18,},
                       'emg':{'emg_model_type':'SVM_PlattScale',
                              'kernel':'rbf',
                              #'svm_C':5.6436, 
                              'svm_C':5.643617263738588,
                              #'gamma':0.0146,
                              'gamma':0.014586498446354922,},
                       'stack_distros':True,
                       })
space_decision.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt'})
decision_scores=[]
for n in range(num_trials):
    results=fuse.function_fuse_withinppt(space_decision)
    decision_scores.append(results)
decision_scores=ml.pd.DataFrame(decision_scores)
print('Mean mean decision accuracy: ',str(decision_scores['fusion_mean_acc'].mean()))



'''feat level selected SEPARATELY'''
space_featlevel=fuse.setup_search_space('featlevel',include_svm=True)
space_featlevel.update({'fusion_alg':'featlevel',
                       'featfuse':
                           {'featfuse_model_type':'SVM_PlattScale', #keep this commented out
                            'kernel':'rbf',#'poly','linear']),
                            #'svm_C':3.6806,
                            'svm_C':3.680576929817659,
                            #'gamma':0.0107,
                            'gamma':0.010688027123522858,
                            },
                       'featfuse_sel_feats_together':False,
                       })
space_featlevel.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt'})
featlevel_scores=[]
for n in range(num_trials):
    results=fuse.function_fuse_withinppt(space_featlevel)
    featlevel_scores.append(results)
featlevel_scores=ml.pd.DataFrame(featlevel_scores)
print('Mean mean featlevel accuracy: ',str(featlevel_scores['fusion_mean_acc'].mean()))



'''feat level selected TOGETHER'''
space_feat_jointsel=fuse.setup_search_space('featlevel',include_svm=True)
space_feat_jointsel.update({'fusion_alg':'featlevel',
                       'featfuse':
                           {'featfuse_model_type':'SVM_PlattScale', #keep this commented out
                            'kernel':'rbf',#'poly','linear']),
                            #'svm_C':97.9778,
                            'svm_C':97.97784331407038,
                            #'gamma':0.0161,
                            'gamma':0.016112099194346027,
                            },
                       'featfuse_sel_feats_together':True,
                       })
space_feat_jointsel.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt'})
feat_joint_sel_scores=[]
for n in range(num_trials):
    results=fuse.function_fuse_withinppt(space_feat_jointsel)
    feat_joint_sel_scores.append(results)
feat_joint_sel_scores=ml.pd.DataFrame(feat_joint_sel_scores)
print('Mean mean featlevel joint selection accuracy: ',str(feat_joint_sel_scores['fusion_mean_acc'].mean()))



space_hierarch=fuse.setup_search_space('hierarchical',include_svm=True)
space_hierarch.update({'fusion_alg':'hierarchical',
                       'eeg':{'eeg_model_type':'kNN',
                              'knn_k':12,},
                       'emg':{'emg_model_type':'SVM_PlattScale',
                              'kernel':'rbf',
                              #'svm_C':4.2807, 
                              'svm_C':4.28074802128932,
                              #'gamma':0.0102,
                              'gamma':0.010215957329744027,},
                       'stack_distros':True,
                       })
space_hierarch.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt'})
hierarch_scores=[]
for n in range(num_trials):
    results=fuse.function_fuse_withinppt(space_hierarch)
    hierarch_scores.append(results)
hierarch_scores=ml.pd.DataFrame(hierarch_scores)
print('Mean mean hierarchical accuracy: ',str(hierarch_scores['fusion_mean_acc'].mean()))



space_inv_hierarch=fuse.setup_search_space('hierarchical_inv',include_svm=True)
space_inv_hierarch.update({'fusion_alg':'hierarchical_inv',
                       'eeg':{'eeg_model_type':'RF',
                              'n_trees':70,},
                       'emg':{'emg_model_type':'SVM_PlattScale',
                              'kernel':'rbf',
                              #'svm_C':18.1983, 
                              'svm_C':18.198340693618345,
                              #'gamma':0.022,
                              'gamma':0.02203231569864118,},
                       'stack_distros':True,
                       })
space_inv_hierarch.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt'})
inv_hierarch_scores=[]
for n in range(num_trials):
    results=fuse.function_fuse_withinppt(space_inv_hierarch)
    inv_hierarch_scores.append(results)
inv_hierarch_scores=ml.pd.DataFrame(inv_hierarch_scores)
print('Mean mean inverse hierarchical accuracy: ',str(inv_hierarch_scores['fusion_mean_acc'].mean()))


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
'''