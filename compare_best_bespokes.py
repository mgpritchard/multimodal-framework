# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 18:03:44 2023

@author: pritcham
"""

import testFusion as fuse
import handleML as ml
import params as params
import pickle
import os
import matplotlib.pyplot as plt
from distro_across_ppts import plot_ppt_distro, plot_ppt_rank, plot_ppt_minmax_normalised
import scipy.stats as stats

test=False
save_overwrite=False
load=True
if test:
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
    space_just_emg.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
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
    space_just_eeg.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
    eeg_scores=[]
    for n in range(num_trials):
        results=fuse.function_fuse_withinppt(space_just_eeg)
        eeg_scores.append(results)
    eeg_scores=ml.pd.DataFrame(eeg_scores)
    print('Mean mean EEG accuracy: ',str(eeg_scores['eeg_mean_acc'].mean()))
    
    
    
    space_decision=fuse.setup_search_space('decision',include_svm=True)
    space_decision.update({'fusion_alg':'lda',
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
    space_decision.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
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
    space_featlevel.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
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
    space_feat_jointsel.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
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
    space_hierarch.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
    hierarch_scores=[]
    for n in range(num_trials):
        results=fuse.function_fuse_withinppt(space_hierarch)
        hierarch_scores.append(results)
    hierarch_scores=ml.pd.DataFrame(hierarch_scores)
    print('Mean mean hierarchical accuracy: ',str(hierarch_scores['fusion_mean_acc'].mean()))
    
    
    
    space_inv_hierarch=fuse.setup_search_space('hierarchical_inv',include_svm=True)
    space_inv_hierarch.update({'fusion_alg':'hierarchical_inv',
                           'eeg':{'eeg_model_type':'RF',
                                  'n_trees':70,
                                  'max_depth':5,},
                           'emg':{'emg_model_type':'SVM_PlattScale',
                                  'kernel':'rbf',
                                  #'svm_C':18.1983, 
                                  'svm_C':18.198340693618345,
                                  #'gamma':0.022,
                                  'gamma':0.02203231569864118,},
                           'stack_distros':True,
                           })
    space_inv_hierarch.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
    inv_hierarch_scores=[]
    for n in range(num_trials):
        results=fuse.function_fuse_withinppt(space_inv_hierarch)
        inv_hierarch_scores.append(results)
    inv_hierarch_scores=ml.pd.DataFrame(inv_hierarch_scores)
    print('Mean mean inverse hierarchical accuracy: ',str(inv_hierarch_scores['fusion_mean_acc'].mean()))

rootpath=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Bespoke_all25\Bespoke_100_compare"
if save_overwrite:
    with open(os.path.join(rootpath,"emg_scores.pkl"),'wb') as f:
        pickle.dump(emg_scores,f)
    with open(os.path.join(rootpath,"eeg_scores.pkl"),'wb') as f:
        pickle.dump(eeg_scores,f)
    with open(os.path.join(rootpath,"decision_scores.pkl"),'wb') as f:
        pickle.dump(decision_scores,f)
    with open(os.path.join(rootpath,"featlevel_scores.pkl"),'wb') as f:
        pickle.dump(featlevel_scores,f)
    with open(os.path.join(rootpath,"featjoint_scores.pkl"),'wb') as f:
        pickle.dump(feat_joint_sel_scores,f)
    with open(os.path.join(rootpath,"hierarch_scores.pkl"),'wb') as f:
        pickle.dump(hierarch_scores,f)
    with open(os.path.join(rootpath,"inv_hierarch_scores.pkl"),'wb') as f:
        pickle.dump(inv_hierarch_scores,f)

if load:    
    with open(os.path.join(rootpath,"emg_scores.pkl"),'rb') as f:
        emg_scores=pickle.load(f)
    with open(os.path.join(rootpath,"eeg_scores.pkl"),'rb') as f:
        eeg_scores=pickle.load(f)
    with open(os.path.join(rootpath,"decision_scores.pkl"),'rb') as f:
        decision_scores=pickle.load(f)
    with open(os.path.join(rootpath,"featlevel_scores.pkl"),'rb') as f:
        featlevel_scores=pickle.load(f)
    with open(os.path.join(rootpath,"featjoint_scores.pkl"),'rb') as f:
        feat_joint_sel_scores=pickle.load(f)
    with open(os.path.join(rootpath,"hierarch_scores.pkl"),'rb') as f:
        hierarch_scores=pickle.load(f)
    with open(os.path.join(rootpath,"inv_hierarch_scores.pkl"),'rb') as f:
        inv_hierarch_scores=pickle.load(f)

allbespoke=emg_scores.join(eeg_scores.join(decision_scores.join(featlevel_scores.join(feat_joint_sel_scores.join(hierarch_scores.join(inv_hierarch_scores,rsuffix='_inv_hierarch'),rsuffix='_hierarch'),rsuffix='_featlevel_joint'),rsuffix='_featlevel'),rsuffix='_decision'),rsuffix='_eeg',lsuffix='_emg')

_,acc_all=plt.subplots()
allbespoke.filter(regex='^fusion_mean_acc.*').boxplot(ax=acc_all)
acc_all.set_title('mean score over 100 repeats')
acc_all.set(xticklabels=['emg\n0.8714','eeg\n0.5492','decision\n0.8642','featlevel\n0.8759','feat_joint\n0.8713','hierarch\n0.8613','inv_hierarch\n0.8510'])

_,acc_no_eeg=plt.subplots()
allbespoke.filter(regex='^(?!.+eeg$)fusion_mean_acc.*').boxplot(ax=acc_no_eeg)
acc_no_eeg.set_ylim(0.84,0.89)
acc_no_eeg.set_title('mean score over 100 repeats')
acc_no_eeg.set(xticklabels=['emg\n0.8714','decision\n0.8642','featlevel\n0.8759','feat_joint\n0.8713','hierarch\n0.8613','inv_hierarch\n0.8510'])

_,acc_eeg=plt.subplots()
allbespoke.filter(regex='^fusion_mean_acc_eeg.*').boxplot(ax=acc_eeg)
acc_eeg.set_ylim(0.52,0.57)
acc_eeg.set_title('mean score over 100 repeats')
acc_eeg.set(xticklabels=['eeg\n0.5492'])

_,t_all=plt.subplots()
allbespoke.filter(regex='^elapsed_time.*').boxplot(ax=t_all)
t_all.set_title('mean time (for all 25 ppts) over 100 repeats')
t_all.set(xticklabels=['emg\n59.58','eeg\n54.76','decision\n76.66','featlevel\n88.85','feat_joint\n112.3','hierarch\n59.00','inv_hierarch\n66.34'])

#somewhat redundant, its the same system design in all 100 runs so the slight randomness shouldnt hugely
#cause one ppt to suddenly be stronger than another
#plot_ppt_distro(emg_scores,'emg_accs',label=' for emg only')
#plot_ppt_distro(featlevel_scores,'fusion_accs',label=' for featlevel separate',rank=True,minmaxnorm=True)
#plot_ppt_distro(eeg_scores,'eeg_accs',label=' for eeg only',rank=True,minmaxnorm=True)

for col in allbespoke.filter(regex='^fusion_accs.*'):
    allbespoke[col.replace('fusion_','stddev_')]=allbespoke.apply(lambda row: ml.np.std(row[col]),axis=1)
    
_,std_all=plt.subplots()
allbespoke.filter(regex='^stddev_.*').boxplot(ax=std_all)
std_all.set_title('stddev across ppts over 100 repeats')
std_all.set(xticklabels=['emg','eeg','decision','featlevel','feat_joint','hierarch','inv_hierarch'])

_,std_no_eeg=plt.subplots()
allbespoke.filter(regex='^(?!.+eeg$)stddev_.*').boxplot(ax=std_no_eeg)
std_no_eeg.set_title('stddev across ppts over 100 repeats')
std_no_eeg.set(xticklabels=['emg','decision','featlevel','feat_joint','hierarch','inv_hierarch'])
    
arch_scores=allbespoke.filter(regex='^(?!.+eeg$)fusion_mean_acc.*')
fvalue, pvalue = stats.f_oneway(arch_scores.iloc[:,0],arch_scores.iloc[:,1],
                                arch_scores.iloc[:,2],arch_scores.iloc[:,3],
                                arch_scores.iloc[:,4],arch_scores.iloc[:,5])
print('anova on all ',fvalue, pvalue)

t_ind, p_ind = stats.ttest_ind(arch_scores.iloc[:,0],arch_scores.iloc[:,2])
print('independent t on emg & featfuse ',t_ind, p_ind)
t_rel, p_rel = stats.ttest_rel(arch_scores.iloc[:,0],arch_scores.iloc[:,2])
print('related t on emg & featfuse ',t_rel, p_rel)


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