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
import pandas as pd

def update_chosen_params(space,arch):
    paramdict={
            'just_emg':{'fusion_alg':'just_emg',
                        'emg':{'emg_model_type':'SVM_PlattScale',
                           'kernel':'rbf',
                           'svm_C':4.172505640055673,
                           'gamma':0.012556011834910268,
                           },
                      },
            'just_eeg':{'fusion_alg':'just_eeg',
                        'eeg':{'eeg_model_type':'LDA',
                           'LDA_solver':'lsqr',
                           'shrinkage':0.037969462143491395,
                           },
                      },
            'decision':{'fusion_alg':'highest_conf',
                      #  'ldafuse':{'LDA_solver':'eigen',
                       #            'shrinkage':0.21678705982755336,},
                        'eeg':{'eeg_model_type':'RF',
                               'n_trees':85,
                               'max_depth':5,},
                        'emg':{'emg_model_type':'SVM_PlattScale',
                               'kernel':'rbf',
                               'svm_C':98.91885185586297,
                               'gamma':0.013119782396855456,},
                        'stack_distros':True,
                        },
            'feat_sep':{'fusion_alg':'featlevel',
                        'featfuse':
                            {'featfuse_model_type':'LDA',
                             'LDA_solver':'svd',
                             'shrinkage':0.6653418849680925,
                             },
                        'featfuse_sel_feats_together':False,
                        },
            'feat_join':{'fusion_alg':'featlevel',
                         'featfuse':
                            {'featfuse_model_type':'LDA',
                             'LDA_solver':'svd',
                             'shrinkage':0.5048542532123359,
                             },
                        'featfuse_sel_feats_together':True,
                        },
            'hierarchical':{'fusion_alg':'hierarchical',
                           'eeg':{'eeg_model_type':'QDA',
                                  'regularisation':0.4558963480892469,},
                           'emg':{'emg_model_type':'SVM_PlattScale',
                                  'kernel':'rbf',
                                  'svm_C':19.403739187394663,
                                  'gamma':0.013797650887036847,},
                           'stack_distros':True,
                           },
            'hierarchical_inv':{'fusion_alg':'hierarchical_inv',
                           'eeg':{'eeg_model_type':'RF',
                                  'n_trees':75,
                                  'max_depth':5,},
                           'emg':{'emg_model_type':'QDA',
                                  'regularisation':0.3324563281128364,},
                           'stack_distros':True,
                           },
            'lit_default_bespoke':{'fusion_alg':'svm',
                                'svmfuse':{'fusesvmPlatt':True,
                                    'svm_C':1.0, #default in sklearn 0.24.2
                                    'kernel':'rbf', #default in sklearn 0.24.2
                                    'gamma':'scale',}, #default in sklearn 0.24.2
                                'eeg':{'eeg_model_type':'LDA',
                                       'LDA_solver':'svd',
                                       'shrinkage':None,
                                       },
                                'emg':{'emg_model_type':'SVM_PlattScale',
                                       'kernel':'rbf',
                                       'svm_C':1.0,
                                       'gamma':'scale',
                                       },
                                'stack_distros':True,
                                },
        }
    space.update(paramdict[arch])
    return space

def test_system(arch,emg,eeg):
    if arch in ['just_emg','just_eeg','decision','hierarchical','hierarchical_inv']:
        space=fuse.setup_search_space(arch,include_svm=True)
    elif arch in ['feat_sep','feat_join']:
        space=fuse.setup_search_space('featlevel',include_svm=True)
        if arch=='feat_sep':
            space.update({'featfuse_sel_feats_together':False})
        elif arch=='feat_join':
            space.update({'featfuse_sel_feats_together':True})
    elif arch== 'lit_default_bespoke':
        space=fuse.setup_search_space('decision',include_svm=True)
    else: raise(ValueError(('Unknown architecture: '+arch)))
    
    space.update({'emg_set':emg,'eeg_set':eeg,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
    space = update_chosen_params(space,arch)
    result=fuse.function_fuse_withinppt(space)
    result.update({'arch':arch})
    return result
    

def test_chosen_bespokes(emg,eeg):
    archs=['just_emg','just_eeg','decision','feat_sep','feat_join','hierarchical','hierarchical_inv']
    results=[]
    for arch in archs:
        for n in range(100):
            '''this doesnt work for some reason, issue with multindex'''
            results.append(pd.DataFrame(test_system(arch,emg,eeg)))
  #      results.append(pd.DataFrame(test_system(arch,emg,eeg)))
#    results=pd.concat(results).set_index(pd.Index(archs))
    results=pd.concat(results).set_index('arch')
    #the commented out bits work with ploitting
    return results

ppt1={'emg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\emg_holdout_ppt1.csv",
      'eeg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\eeg_holdout_ppt1.csv"}
ppt6={'emg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\emg_holdout_ppt6.csv",
      'eeg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\eeg_holdout_ppt6.csv"}
ppt11={'emg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\emg_holdout_ppt11.csv",
      'eeg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\eeg_holdout_ppt11.csv"}
ppt16={'emg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\emg_holdout_ppt16.csv",
      'eeg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\eeg_holdout_ppt16.csv"}
ppt21={'emg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\emg_holdout_ppt21.csv",
      'eeg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\eeg_holdout_ppt21.csv"}

holdout_ppts=[ppt1,ppt6,ppt11,ppt16,ppt21]

ppt_scores_list=[]
for ppt in holdout_ppts:
    emg=pd.read_csv(ppt['emg_path'],delimiter=',')
    eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
    emg,eeg=fuse.balance_set(emg,eeg)
    results=test_chosen_bespokes(emg,eeg)
    ppt_scores_list.append(results)
ppt_scores=pd.concat(ppt_scores_list, axis=0, keys=['ppt1','ppt6','ppt11','ppt16','ppt21'])
ppt_scores=ppt_scores.swaplevel(-2,-1)
ppt_scores.index.names=['arch','ppt']

if 1:
    ppt_scores.reset_index(drop=False)
    ppt_scores.to_csv(r"C:\Users\pritcham\Documents\RQ1_plots_stats\RQ1_R_stats\bespoke_100_repeats.csv")
   # bigDF.to_csv(r"C:\Users\pritcham\Documents\RQ1_plots_stats\RQ1_R_stats\bespoke_Meansof100reps.csv")
    raise

ppt_scores=pd.read_csv(r"C:\Users\pritcham\Documents\RQ1_plots_stats\RQ1_R_stats\bespoke_Meansof100reps.csv")
ppt_scores=ppt_scores.set_index('ppt')
groupedBespUM=ppt_scores.loc[:,['just_eeg','just_emg']].T
groupedBespFus=ppt_scores.loc[:,['decision','feat_join','feat_sep','hierarch','inv_hierarch']].T

_,acc_groupedBarsFus=plt.subplots()
#groupedFus=groupedBespFus.reset_index().loc[:,['ppt','fusion_acc']]
#groupedFus=groupedFus.reset_index()
#bigDF.pivot(index='arch',columns='ppt')['fusion_acc'] #same as below
#https://stackoverflow.com/questions/22127569/opposite-of-melt-in-python-pandas
#groupedFus=groupedFus.pivot(*groupedFus)
groupedBespFus.plot(kind='bar',ax=acc_groupedBarsFus,ylim=(0.7,0.95),rot=0)
acc_groupedBarsFus.set(xlabel='Architecture')
# Shrink current axis's height by 10% on the top
#https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
#  box = acc_groupedBarsFus.get_position()
#  acc_groupedBarsFus.set_position([box.x0, box.y0 - box.height * 0.1,
#                   box.width, box.height * 0.9])
# Put a legend above current axis
acc_groupedBarsFus.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
      fancybox=False, shadow=False, ncol=5)
#acc_groupedBarsFus.legend(loc='best',ncol=5)

  
_,acc_groupedBarsUM=plt.subplots()
#groupedUM=groupedBespUM.reset_index(level="ppt").loc[:,['ppt','fusion_acc']]
#groupedUM=groupedUM.reset_index()
#bigDF.pivot(index='arch',columns='ppt')['fusion_acc'] #same as below
#https://stackoverflow.com/questions/22127569/opposite-of-melt-in-python-pandas
#groupedUM=groupedUM.pivot(*groupedUM)
groupedBespUM.plot(kind='bar',ax=acc_groupedBarsUM,ylim=(0.5,0.95),rot=0)
acc_groupedBarsUM.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)








scores1=ppt_scores.xs('ppt1',level='ppt')
scores6=ppt_scores.xs('ppt6',level='ppt')
scores11=ppt_scores.xs('ppt11',level='ppt')
scores16=ppt_scores.xs('ppt16',level='ppt')
scores21=ppt_scores.xs('ppt21',level='ppt')

scores_emg=ppt_scores.xs('just_emg',level='arch')
scores_eeg=ppt_scores.xs('just_eeg',level='arch')
scores_dec=ppt_scores.xs('decision',level='arch')
scores_featsep=ppt_scores.xs('feat_sep',level='arch')
scores_featjoin=ppt_scores.xs('feat_join',level='arch')
scores_hierarch=ppt_scores.xs('hierarchical',level='arch')
scores_inv_hierarch=ppt_scores.xs('hierarchical_inv',level='arch')


#allbespoke=emg_scores.join(eeg_scores.join(decision_scores.join(featlevel_scores.join(feat_joint_sel_scores.join(hierarch_scores.join(inv_hierarch_scores,rsuffix='_inv_hierarch'),rsuffix='_hierarch'),rsuffix='_featlevel_joint'),rsuffix='_featlevel'),rsuffix='_decision'),rsuffix='_eeg',lsuffix='_emg')
#for col in allbespoke.filter(regex='^fusion_accs.*'):
#    allbespoke[col.replace('fusion_','stddev_')]=allbespoke.apply(lambda row: ml.np.std(row[col]),axis=1)

_,acc_all=plt.subplots()
#allbespoke.filter(regex='^fusion_mean_acc.*').boxplot(ax=acc_all)
ppt_scores.boxplot(column=['fusion_mean_acc'],by='arch',ax=acc_all)
acc_all.set_title('mean score over 5 heldout ppts')
acc_all.set(xticklabels=['emg\n0.8778','eeg\n0.5480','decision\n0.8783','featlevel\n0.8613','feat_joint\n0.8648','hierarch\n0.8898','inv_hierarch\n0.8368'])
acc_all.set(xlabel='Architecture, below: peak accuracy reached in optimisation')

_,acc_noEEG=plt.subplots()
#allbespoke.filter(regex='^fusion_mean_acc.*').boxplot(ax=acc_all)
ppt_scores.loc[~ppt_scores.index.isin(['just_eeg'],level='arch')].boxplot(column=['fusion_mean_acc'],by='arch',ax=acc_noEEG)
acc_noEEG.get_figure().suptitle('mean score over 5 heldout ppts')
acc_noEEG.set_title('')
acc_noEEG.set(xticklabels=['emg\n0.8778','decision\n0.8783','featlevel\n0.8613','feat_joint\n0.8648','hierarch\n0.8898','inv_hierarch\n0.8368'])
acc_noEEG.set(xlabel='Architecture, below: peak accuracy reached in optimisation')

_,acc_noEEG_lines=plt.subplots()
scores1.loc[~scores1.index.isin(['just_eeg'],level='arch')].plot(y='fusion_mean_acc',ax=acc_noEEG_lines)#,style='o')
scores6.loc[~scores1.index.isin(['just_eeg'],level='arch')].plot(y='fusion_mean_acc',ax=acc_noEEG_lines)#,style='o')
scores11.loc[~scores1.index.isin(['just_eeg'],level='arch')].plot(y='fusion_mean_acc',ax=acc_noEEG_lines)#,style='o')
scores16.loc[~scores1.index.isin(['just_eeg'],level='arch')].plot(y='fusion_mean_acc',ax=acc_noEEG_lines)#,style='o')
scores21.loc[~scores1.index.isin(['just_eeg'],level='arch')].plot(y='fusion_mean_acc',ax=acc_noEEG_lines)#,style='o')
acc_noEEG_lines.get_figure().suptitle('')
acc_noEEG_lines.set_title('')
acc_noEEG_lines.legend(['ppt1','ppt6','ppt11','ppt16','ppt21'])
acc_noEEG_lines.set(xlabel='Architecture')

_,t_all=plt.subplots()
#allbespoke.filter(regex='^fusion_mean_acc.*').boxplot(ax=acc_all)
ppt_scores.boxplot(column=['elapsed_time'],by='arch',ax=t_all)
t_all.get_figure().suptitle('mean score over 5 heldout ppts')
t_all.set(xticklabels=['emg','eeg','decision','featlevel','feat_joint','hierarch','inv_hierarch'])
t_all.set(xlabel='Total train & pred time')

all_accs=pd.DataFrame()
for (archname, res) in zip(['just_emg','decision','feat_sep','feat_join','hierarchical','hierarchical_inv'],
                         [scores_emg,scores_dec,scores_featsep,scores_featjoin,scores_hierarch,scores_inv_hierarch]):
    all_accs[archname]=res['fusion_mean_acc']

'''    
fvalue, pvalue = stats.f_oneway(all_accs.iloc[:,0],all_accs.iloc[:,1],
                                all_accs.iloc[:,2],all_accs.iloc[:,3],
                                all_accs.iloc[:,4],all_accs.iloc[:,5])
print('anova on all ',fvalue, pvalue)

t_ind, p_ind = stats.ttest_ind(all_accs.iloc[:,0],all_accs.iloc[:,4])
print('independent t on emg & hierarch ',t_ind, p_ind)
#t_rel, p_rel = stats.ttest_rel(all_accs.iloc[:,0],all_accs.iloc[:,2])
#print('related t on emg & hierarch ',t_rel, p_rel)
t_ind, p_ind = stats.ttest_ind(all_accs.iloc[:,0],all_accs.iloc[:,2])
print('independent t on emg & decision ',t_ind, p_ind)
'''
