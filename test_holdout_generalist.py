# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 18:03:44 2023

@author: pritcham
"""

import testFusion as fuse
import handleML as ml
import handleFeats as feats
import params as params
import pickle
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from distro_across_ppts import plot_ppt_distro, plot_ppt_rank, plot_ppt_minmax_normalised
import scipy.stats as stats
import pandas as pd
import numpy as np
import time
import statsmodels.api as sm
from statsmodels.formula.api import ols

def update_chosen_params(space,arch):
    paramdict={
            'just_emg':{'fusion_alg':'just_emg',
                        'emg':{'emg_model_type':'LDA',
                           'LDA_solver':'eigen',
                           'shrinkage':0.07440592720562522,
                           },
                      },
            'just_eeg':{'fusion_alg':'just_eeg',
                        'eeg':{'eeg_model_type':'LDA',
                           'LDA_solver':'lsqr',
                           'shrinkage':0.043549089484270026,
                           },
                      },    
            'decision':{'fusion_alg':'svm',
                        'svmfuse':{'svm_C':0.05380895830748056,},
                        'eeg':{'eeg_model_type':'LDA',
                               'LDA_solver':'eigen',
                               'shrinkage':0.3692791355027271,},
                        'emg':{'emg_model_type':'LDA',
                               'LDA_solver':'lsqr',
                               'shrinkage':0.23494163949824712,},
                        'stack_distros':True,
                        },
            'hierarchical':{'fusion_alg':'hierarchical',
                       'eeg':{'eeg_model_type':'LDA',
                               'LDA_solver':'svd',
                               'shrinkage':0.09831710168084823,},
                       'emg':{'emg_model_type':'LDA',
                               'LDA_solver':'svd',
                               'shrinkage':0.9411890733402433,},
                           'stack_distros':True,
                           },
            'hierarchical_inv':{'fusion_alg':'hierarchical_inv',
                       'eeg':{'eeg_model_type':'LDA',
                               'LDA_solver':'svd',
                               'shrinkage':0.007751611941250992,},
                        'emg':{'emg_model_type':'LDA',
                               'LDA_solver':'lsqr',
                               'shrinkage':0.02289551604694198,},
                           'stack_distros':True,
                           },
            'feat_sep':{'fusion_alg':'featlevel',
                        'featfuse':
                            {'featfuse_model_type':'LDA',
                             'LDA_solver':'eigen',
                             'shrinkage':0.023498953661387587,
                             },
                        'featfuse_sel_feats_together':False,
                        },
            'feat_join':{'fusion_alg':'featlevel',
                        'featfuse':
                            {'featfuse_model_type':'LDA',
                             'LDA_solver':'lsqr',
                             'shrinkage':0.18709819935238686,
                             },
                        'featfuse_sel_feats_together':True,
                        },
            'lit_default_generalist':{'fusion_alg':'mean',
                                'eeg':{'eeg_model_type':'LDA',
                                       'LDA_solver':'svd', #default in sklearn 0.24.2
                                       'shrinkage':None, #default in sklearn 0.24.2
                                       },
                    #            'emg':{'emg_model_type':'RF',  #trying LINEAR kernel svm as per Tryon2019 generalist
                    #                   'n_trees':100, #default in sklearn 0.24.2
                    #                   'max_depth':None, #default in sklearn 0.24.2
                    #                   },
                                'emg':{'emg_model_type':'SVM_PlattScale',
                                       'kernel':'linear',
                                       'svm_C':1.0,
                                       'gamma':None,
                                       },
                                'stack_distros':True,
                                },
        }
    space.update(paramdict[arch])
    return space

def fuse_LOO(emg_others,eeg_others,emg_ppt,eeg_ppt,args):
    start=time.time()
    
    if args['scalingtype']:
        emg_others,emgscaler=feats.scale_feats_train(emg_others,args['scalingtype'])
        eeg_others,eegscaler=feats.scale_feats_train(eeg_others,args['scalingtype'])
        emg_ppt=feats.scale_feats_test(emg_ppt,emgscaler)
        eeg_ppt=feats.scale_feats_test(eeg_ppt,eegscaler)
        
    if args['fusion_alg']=='svm':
            
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.fusion_SVM(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
    
    elif args['fusion_alg']=='lda':
            
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.fusion_LDA(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
        
    elif args['fusion_alg']=='rf':
            
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.fusion_RF(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
   
    elif args['fusion_alg']=='hierarchical':
        
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.fusion_hierarchical(emg_others, eeg_others, emg_ppt, eeg_ppt, args)

    elif args['fusion_alg']=='hierarchical_inv':          
                                              
        if not args['get_train_acc']:    
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.fusion_hierarchical_inv(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
        else:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=fuse.fusion_hierarchical_inv(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
             
    elif args['fusion_alg']=='featlevel':  
                        
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.feature_fusion(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
    
    elif args['fusion_alg']=='just_emg':
        
        if not args['get_train_acc']:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.only_EMG(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
        else:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=fuse.only_EMG(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
    
    elif args['fusion_alg']=='just_eeg':
        
        if not args['get_train_acc']:    
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.only_EEG(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
        else:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=fuse.only_EEG(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
                
    else:
                    
        emg_others=ml.drop_ID_cols(emg_others)
        eeg_others=ml.drop_ID_cols(eeg_others)
        
        if args['trialmode']=='WithinPpt':
            sel_cols_eeg=feats.sel_feats_l1_df(eeg_others,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
            #sel_cols_eeg=feats.sel_percent_feats_df(eeg_others,percent=3)
            sel_cols_eeg=np.append(sel_cols_eeg,eeg_others.columns.get_loc('Label'))
            
            sel_cols_emg=feats.sel_percent_feats_df(emg_others,percent=15)
            sel_cols_emg=np.append(sel_cols_emg,emg_others.columns.get_loc('Label'))
        elif args['trialmode']=='LOO':
            pptidx=int(emg_ppt['ID_pptID'].iloc[0])-1
            sel_cols_eeg=[eeg_others.columns.get_loc(col) for col in args['eeg_feats_LOO'].iloc[pptidx].tolist()]
            sel_cols_emg=[emg_others.columns.get_loc(col) for col in args['emg_feats_LOO'].iloc[pptidx].tolist()]
            
        eeg_others=eeg_others.iloc[:,sel_cols_eeg]
        emg_others=emg_others.iloc[:,sel_cols_emg]
        
        emg_model,eeg_model=fuse.train_models_opt(emg_others,eeg_others,args)
    
        classlabels = emg_model.classes_
        
        emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            
        targets, predlist_emg, predlist_eeg, predlist_fusion,_,_,_= fuse.refactor_synced_predict(emg_ppt, eeg_ppt, emg_model, eeg_model, classlabels,args, sel_cols_eeg,sel_cols_emg)

    gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels=fuse.classes_from_preds(targets,predlist_emg,predlist_eeg,predlist_fusion,classlabels)
    '''could calculate log loss if got the probabilities back''' #https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
    
    if args['plot_confmats']:
        gesturelabels=[params.idx_to_gestures[label] for label in classlabels]
        fuse.tt.confmat(gest_truth,gest_pred_eeg,gesturelabels,title=(args['conftitle']+'EEG'))
        fuse.tt.confmat(gest_truth,gest_pred_emg,gesturelabels,title='EMG')
        fuse.tt.confmat(gest_truth,gest_pred_fusion,gesturelabels,title='Fusion')
        
    emg_acc=(fuse.accuracy_score(gest_truth,gest_pred_emg))
    eeg_acc=(fuse.accuracy_score(gest_truth,gest_pred_eeg))
    acc=(fuse.accuracy_score(gest_truth,gest_pred_fusion))
    
    kappa=(fuse.cohen_kappa_score(gest_truth,gest_pred_fusion))
    
    if args['get_train_acc']:
        train_truth=[params.idx_to_gestures[gest] for gest in traintargs]
        train_preds=[params.idx_to_gestures[pred] for pred in predlist_train]
        train_acc=(fuse.accuracy_score(train_truth,train_preds))
    else:
        train_acc=(0)

    end=time.time()
    
    if 'getPreds' in args:
        if args['getPreds']==True:
            return {
                'loss': 1-acc,
                'kappa':kappa,
                'fusion_acc':acc,
                'emg_acc':emg_acc,
                'eeg_acc':eeg_acc,
                'train_acc':train_acc,
                'elapsed_time':end-start,
                'gest_truth':gest_truth,
                'gest_preds':gest_pred_fusion,}
        else:
            return {
                'loss': 1-acc,
                'kappa':kappa,
                'fusion_acc':acc,
                'emg_acc':emg_acc,
                'eeg_acc':eeg_acc,
                'train_acc':train_acc,
                'elapsed_time':end-start,}
    else:
        return {
            'loss': 1-acc,
            'kappa':kappa,
            'fusion_acc':acc,
            'emg_acc':emg_acc,
            'eeg_acc':eeg_acc,
            'train_acc':train_acc,
            'elapsed_time':end-start,}

def get_confmats_eeg():
    trainEEGpath=params.jeong_eeg_noholdout
    trainEMGpath=params.jeong_emg_noholdout
    
    trainEEG=pd.read_csv(trainEEGpath)
    trainEMG=pd.read_csv(trainEMGpath)
    trainEMG,trainEEG=fuse.balance_set(trainEMG,trainEEG)
    
    
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
    
    
    space=fuse.setup_search_space('just_eeg',include_svm=False)
    space=update_chosen_params(space,'just_eeg')
    space.update({'trialmode':'LOO'})
    emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
    eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
    space.update({'emg_feats_LOO':emg_cols,
                  'eeg_feats_LOO':eeg_cols,})
    
    space.update({'plot_confmats':True})
    space.update({'getPreds':True})
    
    ppt_scores=[]
    for ppt in holdout_ppts:
        emg=pd.read_csv(ppt['emg_path'],delimiter=',')
        eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
        emg,eeg=fuse.balance_set(emg,eeg)
        pptid = ppt['emg_path'].split('_')[-1][:-4]
        space.update({'conftitle':(pptid+' Generalist ')})
        results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
        ppt_scores.append(results)
        
    ppt_scores_just_eeg=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
    
    _,acc_eeg=plt.subplots()
    ppt_scores_just_eeg.boxplot(column='fusion_acc',ax=acc_eeg)
    acc_eeg.set(ylim=([0, 1]))
    
    scoremean=np.mean(ppt_scores_just_eeg['fusion_acc'])
    scorestd=np.std(ppt_scores_just_eeg['fusion_acc'])
    print('Mean eeg acc over 5 heldout: ',str(scoremean))
    print('Std dev eeg acc over 5 heldout: ',str(scorestd))
    
    return ppt_scores_just_eeg

if __name__ == '__main__':
    
    #eegScores = get_confmats_eeg()
    '''could do again but getting preds back for merged ConfMat?'''
    ''' will NEED to do again to get per ppt accs etc'''
    #raise
    
    test_archs=False
    save_overwrite_scores=False
    load_scores=False
    
    test_litDefault=False
    
    if test_litDefault:
        trainEEGpath=params.jeong_eeg_noholdout
        trainEMGpath=params.jeong_emg_noholdout
        
        trainEEG=pd.read_csv(trainEEGpath)
        trainEMG=pd.read_csv(trainEMGpath)
        trainEMG,trainEEG=fuse.balance_set(trainEMG,trainEEG)
              
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
        
        
        space=fuse.setup_search_space('decision',include_svm=True)
        space=update_chosen_params(space, 'lit_default_generalist')
        
        space.update({'trialmode':'LOO'})
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,})
        
        ppt_scores=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
            ppt_scores.append(results)
            
        ppt_scores_lit_def=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        
        _,acc_lit_def=plt.subplots()
        ppt_scores_lit_def.boxplot(column='fusion_acc',ax=acc_lit_def)
        acc_lit_def.set(ylim=([0, 1]))
        
        scoremean=np.mean(ppt_scores_lit_def['fusion_acc'])
        scorestd=np.std(ppt_scores_lit_def['fusion_acc'])
        print('Mean feat join acc over 5 heldout: ',str(scoremean))
        print('Std dev feat join acc over 5 heldout: ',str(scorestd))
        
        rootpath=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\GeneralistOnHoldout" 
        with open(os.path.join(rootpath,"lit_def_scores.pkl"),'wb') as f:
            pickle.dump(ppt_scores_lit_def,f)
            
        raise
    
    if test_archs:
        trainEEGpath=params.jeong_eeg_noholdout
        trainEMGpath=params.jeong_emg_noholdout
        
        trainEEG=pd.read_csv(trainEEGpath)
        trainEMG=pd.read_csv(trainEMGpath)
        trainEMG,trainEEG=fuse.balance_set(trainEMG,trainEEG)
        
        
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
        
        
        
        space=fuse.setup_search_space('just_eeg',include_svm=False)
        
        space=update_chosen_params(space,'just_eeg')
        space.update({'trialmode':'LOO'})
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,})
        
        ppt_scores=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
            ppt_scores.append(results)
            
        ppt_scores_just_eeg=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        
        _,acc_eeg=plt.subplots()
        ppt_scores_just_eeg.boxplot(column='fusion_acc',ax=acc_eeg)
        acc_eeg.set(ylim=([0, 1]))
        
        scoremean=np.mean(ppt_scores_just_eeg['fusion_acc'])
        scorestd=np.std(ppt_scores_just_eeg['fusion_acc'])
        print('Mean eeg acc over 5 heldout: ',str(scoremean))
        print('Std dev eeg acc over 5 heldout: ',str(scorestd))
        
        
        
        raise
        
        
        
        
        
        space=fuse.setup_search_space('featlevel',include_svm=False)
        
        space=update_chosen_params(space,'feat_join')
        space.update({'trialmode':'LOO'})
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        emgeegcols=pd.read_csv(params.jointemgeegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,
                      'jointemgeeg_feats_LOO':emgeegcols,})
        
        ppt_scores=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
            ppt_scores.append(results)
            
        ppt_scores_feat_join=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        
        _,acc_feat_join=plt.subplots()
        ppt_scores_feat_join.boxplot(column='fusion_acc',ax=acc_feat_join)
        acc_feat_join.set(ylim=([0, 1]))
        
        scoremean=np.mean(ppt_scores_feat_join['fusion_acc'])
        scorestd=np.std(ppt_scores_feat_join['fusion_acc'])
        print('Mean feat join acc over 5 heldout: ',str(scoremean))
        print('Std dev feat join acc over 5 heldout: ',str(scorestd))
        
        
        
        
        space=fuse.setup_search_space('featlevel',include_svm=False)
        
        space=update_chosen_params(space,'feat_sep')
        space.update({'trialmode':'LOO'})
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,})
        
        ppt_scores=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
            ppt_scores.append(results)
            
        ppt_scores_feat_sep=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        
        _,acc_feat_sep=plt.subplots()
        ppt_scores_feat_sep.boxplot(column='fusion_acc',ax=acc_feat_sep)
        acc_feat_sep.set(ylim=([0, 1]))
        
        scoremean=np.mean(ppt_scores_feat_sep['fusion_acc'])
        scorestd=np.std(ppt_scores_feat_sep['fusion_acc'])
        print('Mean feat sep acc over 5 heldout: ',str(scoremean))
        print('Std dev feat sep acc over 5 heldout: ',str(scorestd))
        
        
        
        
        space=fuse.setup_search_space('decision',include_svm=False)
        
        space=update_chosen_params(space,'decision')
        space.update({'trialmode':'LOO'})
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,})
        
        ppt_scores=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
            ppt_scores.append(results)
            
        ppt_scores_dec=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        
        _,acc_dec=plt.subplots()
        ppt_scores_dec.boxplot(column='fusion_acc',ax=acc_dec)
        acc_dec.set(ylim=([0, 1]))
        
        scoremean=np.mean(ppt_scores_dec['fusion_acc'])
        scorestd=np.std(ppt_scores_dec['fusion_acc'])
        print('Mean decision acc over 5 heldout: ',str(scoremean))
        print('Std dev decision acc over 5 heldout: ',str(scorestd))
        
        
        
        
        space=fuse.setup_search_space('just_emg',include_svm=False)
        
        space=update_chosen_params(space,'just_emg')
        space.update({'trialmode':'LOO'})
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,})
        
        ppt_scores=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
            ppt_scores.append(results)
            
        ppt_scores_just_emg=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        
        _,acc_emg=plt.subplots()
        ppt_scores_just_emg.boxplot(column='fusion_acc',ax=acc_emg)
        acc_emg.set(ylim=([0, 1]))
        
        scoremean=np.mean(ppt_scores_just_emg['fusion_acc'])
        scorestd=np.std(ppt_scores_just_emg['fusion_acc'])
        print('Mean emg acc over 5 heldout: ',str(scoremean))
        print('Std dev emg acc over 5 heldout: ',str(scorestd))
        
        
        
        
        
        
        
        
        
        space=fuse.setup_search_space('hierarchical',include_svm=False)
        
        space=update_chosen_params(space,'hierarchical')
        space.update({'trialmode':'LOO'})
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,})
        
        ppt_scores=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
            ppt_scores.append(results)
            
        ppt_scores_hierarch=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        
        _,acc_hierarch=plt.subplots()
        ppt_scores_hierarch.boxplot(column='fusion_acc',ax=acc_hierarch)
        acc_hierarch.set(ylim=([0, 1]))
        
        scoremean=np.mean(ppt_scores_hierarch['fusion_acc'])
        scorestd=np.std(ppt_scores_hierarch['fusion_acc'])
        print('Mean hierarch acc over 5 heldout: ',str(scoremean))
        print('Std dev hierarch acc over 5 heldout: ',str(scorestd))
        
        
        
        space=fuse.setup_search_space('hierarchical_inv',include_svm=False)
        
        space=update_chosen_params(space,'hierarchical_inv')
        space.update({'trialmode':'LOO'})
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,})
        
        ppt_scores=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
            ppt_scores.append(results)
            
        ppt_scores_inv_hierarch=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        
        _,acc_invh=plt.subplots()
        ppt_scores_inv_hierarch.boxplot(column='fusion_acc',ax=acc_invh)
        acc_invh.set(ylim=([0, 1]))
        
        scoremean=np.mean(ppt_scores_inv_hierarch['fusion_acc'])
        scorestd=np.std(ppt_scores_inv_hierarch['fusion_acc'])
        print('Mean inv hierarch acc over 5 heldout: ',str(scoremean))
        print('Std dev inv hierarch acc over 5 heldout: ',str(scorestd))
    
    rootpath=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\GeneralistOnHoldout" 
    if save_overwrite_scores:
        with open(os.path.join(rootpath,"emg_scores.pkl"),'wb') as f:
            pickle.dump(ppt_scores_just_emg,f)
        with open(os.path.join(rootpath,"eeg_scores.pkl"),'wb') as f:
            pickle.dump(ppt_scores_just_eeg,f)
        with open(os.path.join(rootpath,"decision_scores.pkl"),'wb') as f:
            pickle.dump(ppt_scores_dec,f)
        with open(os.path.join(rootpath,"featsep_scores.pkl"),'wb') as f:
            pickle.dump(ppt_scores_feat_sep,f)
        with open(os.path.join(rootpath,"featjoint_scores.pkl"),'wb') as f:
            pickle.dump(ppt_scores_feat_join,f)
        with open(os.path.join(rootpath,"hierarch_scores.pkl"),'wb') as f:
            pickle.dump(ppt_scores_hierarch,f)
        with open(os.path.join(rootpath,"inv_hierarch_scores.pkl"),'wb') as f:
            pickle.dump(ppt_scores_inv_hierarch,f)
    
    if load_scores:    
        with open(os.path.join(rootpath,"emg_scores.pkl"),'rb') as f:
            ppt_scores_just_emg=pickle.load(f)
        with open(os.path.join(rootpath,"eeg_scores.pkl"),'rb') as f:
            ppt_scores_just_eeg=pickle.load(f)
        with open(os.path.join(rootpath,"decision_scores.pkl"),'rb') as f:
            ppt_scores_dec=pickle.load(f)
        with open(os.path.join(rootpath,"featsep_scores.pkl"),'rb') as f:
            ppt_scores_feat_sep=pickle.load(f)
        with open(os.path.join(rootpath,"featjoint_scores.pkl"),'rb') as f:
            ppt_scores_feat_join=pickle.load(f)
        with open(os.path.join(rootpath,"hierarch_scores.pkl"),'rb') as f:
            ppt_scores_hierarch=pickle.load(f)
        with open(os.path.join(rootpath,"inv_hierarch_scores.pkl"),'rb') as f:
            ppt_scores_inv_hierarch=pickle.load(f)
    
    ppt_scores_all=[ppt_scores_just_eeg,ppt_scores_just_emg,ppt_scores_dec,ppt_scores_feat_sep,ppt_scores_feat_join,ppt_scores_hierarch,ppt_scores_inv_hierarch]
    ppt_scores_all=pd.concat(ppt_scores_all,axis=0,keys=['just_eeg','just_emg','decision','feat_sep','feat_join','hierarch','inv_hierarch'])
    ppt_scores_all.index.names=['arch','ppt']
    
    scores1=ppt_scores_all.xs('ppt1',level='ppt')
    scores6=ppt_scores_all.xs('ppt6',level='ppt')
    scores11=ppt_scores_all.xs('ppt11',level='ppt')
    scores16=ppt_scores_all.xs('ppt16',level='ppt')
    scores21=ppt_scores_all.xs('ppt21',level='ppt')

    
    
    _,acc_noEEG=plt.subplots()
    scores1.loc[~scores1.index.isin(['just_eeg'],level='arch')].plot(y='fusion_acc',ax=acc_noEEG)#,style='o')
    scores6.loc[~scores1.index.isin(['just_eeg'],level='arch')].plot(y='fusion_acc',ax=acc_noEEG)#,style='o')
    scores11.loc[~scores1.index.isin(['just_eeg'],level='arch')].plot(y='fusion_acc',ax=acc_noEEG)#,style='o')
    scores16.loc[~scores1.index.isin(['just_eeg'],level='arch')].plot(y='fusion_acc',ax=acc_noEEG)#,style='o')
    scores21.loc[~scores1.index.isin(['just_eeg'],level='arch')].plot(y='fusion_acc',ax=acc_noEEG)#,style='o')
    acc_noEEG.get_figure().suptitle('')
    acc_noEEG.set_title('')
    acc_noEEG.legend(['ppt1','ppt6','ppt11','ppt16','ppt21'])
    acc_noEEG.set(xlabel='Architecture')
    
    accuracies=ppt_scores_all[~ppt_scores_all.index.isin(['just_eeg'],level='arch')]['fusion_acc'].reset_index(drop=False)
    
    accuracies_withEEG=ppt_scores_all['fusion_acc'].reset_index(drop=False)
    
    
    _,acc_groupedBars=plt.subplots()
    bigDF=ppt_scores_all.reset_index(level="ppt").loc[:,['ppt','fusion_acc']]
    bigDF=bigDF.reset_index()
    #bigDF.pivot(index='arch',columns='ppt')['fusion_acc'] #same as below
    #https://stackoverflow.com/questions/22127569/opposite-of-melt-in-python-pandas
    bigDF=bigDF.pivot(*bigDF)
    bigDF.plot(kind='bar',ax=acc_groupedBars,ylim=(0.45,0.85))
    
    _,acc_groupedbyPpt=plt.subplots()
    bigDF=ppt_scores_all.reset_index(level="arch").loc[:,['arch','fusion_acc']]
    bigDF=bigDF.reset_index()
    #bigDF.pivot(index='arch',columns='ppt')['fusion_acc'] #same as below
    #https://stackoverflow.com/questions/22127569/opposite-of-melt-in-python-pandas
    bigDF=bigDF.pivot(*bigDF)
    bigDF.plot(kind='bar',ax=acc_groupedbyPpt,ylim=(0.45,0.85))
    
    
    
    ppt_scores_fus=[ppt_scores_dec,ppt_scores_feat_sep,ppt_scores_feat_join,ppt_scores_hierarch,ppt_scores_inv_hierarch]
    ppt_scores_fus=pd.concat(ppt_scores_fus,axis=0,keys=['decision','feat_sep','feat_join','hierarch','inv_hierarch'])
    ppt_scores_fus.index.names=['arch','ppt']
    
    ppt_scores_unimodal=[ppt_scores_just_eeg,ppt_scores_just_emg]
    ppt_scores_unimodal=pd.concat(ppt_scores_unimodal,axis=0,keys=['just_eeg','just_emg'])
    ppt_scores_unimodal.index.names=['arch','ppt']
    
    _,acc_groupedBarsFus=plt.subplots()
    groupedFus=ppt_scores_fus.reset_index(level="ppt").loc[:,['ppt','fusion_acc']]
    groupedFus=groupedFus.reset_index()
    #bigDF.pivot(index='arch',columns='ppt')['fusion_acc'] #same as below
    #https://stackoverflow.com/questions/22127569/opposite-of-melt-in-python-pandas
    groupedFus=groupedFus.pivot(*groupedFus)
    groupedFus.plot(kind='bar',ax=acc_groupedBarsFus,ylim=(0.6,0.85),rot=0)
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
    groupedUM=ppt_scores_unimodal.reset_index(level="ppt").loc[:,['ppt','fusion_acc']]
    groupedUM=groupedUM.reset_index()
    #bigDF.pivot(index='arch',columns='ppt')['fusion_acc'] #same as below
    #https://stackoverflow.com/questions/22127569/opposite-of-melt-in-python-pandas
    groupedUM=groupedUM.pivot(*groupedUM)
    groupedUM.plot(kind='bar',ax=acc_groupedBarsUM,ylim=(0.45,0.8),rot=0)
    acc_groupedBarsUM.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
    
    # Performing two-way ANOVA
   # model = ols('fusion_acc ~ C(arch) + C(ppt) + C(arch):C(ppt)', data=accuracies).fit()
   # anovaresult=sm.stats.anova_lm(model, typ=2)
   # print(anovaresult)
    
 #   accuracies.to_csv(r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\GeneralistOnHoldout\accuracies_noEEG.csv")
 #   accuracies_withEEG.to_csv(r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\GeneralistOnHoldout\accuracies_withEEG.csv")


