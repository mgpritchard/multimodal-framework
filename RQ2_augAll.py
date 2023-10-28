# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:53:04 2023

@author: pritcham
"""
from RQ2_opt import get_ppt_split_flexi, balance_set, fusion_SVM, fusion_LDA, fusion_RF, train_models_opt, refactor_synced_predict, classes_from_preds, setup_search_space, inspect_set_balance
import os
import sys
import numpy as np
import statistics as stats
import handleDatawrangle as wrangle
import handleFeats as feats
import handleML as ml
import handleComposeDataset as comp
import handleTrainTestPipeline as tt
import handleFusion as fusion
import params
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, log_loss, confusion_matrix, ConfusionMatrixDisplay #plot_confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.patches import Ellipse
from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, Trials
from hyperopt.pyll import scope, stochastic
import time
import pandas as pd
import pickle as pickle

def fuse_fullbespoke(args):
    start=time.time()
    if not args['data_in_memory']:
        raise ValueError('No data in memory')
    else:
        emg_ppt=args['emg_set']
        eeg_ppt=args['eeg_set']
    if not args['prebalanced']: 
        raise ValueError('Data not balanced')

    
    if not emg_ppt['ID_stratID'].equals(eeg_ppt['ID_stratID']):
        raise ValueError('EMG & EEG performances misaligned')
    gest_perfs=emg_ppt['ID_stratID'].unique()
    gest_strat=pd.DataFrame([gest_perfs,[perf.split('.')[1][-1] for perf in gest_perfs]]).transpose()
    
    random_split=random.randint(0,100)
    train_split,test_split=train_test_split(gest_strat,test_size=args['testset_size'],
                                            random_state=random_split,stratify=gest_strat[1])
    if args['opt_method']=='subject':
        eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split[0])]
        emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]
    elif args['opt_method']=='non-subject aug':
        subj=args['subject-id']
        eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split[0])]
        emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]
        eeg_test=eeg_test[eeg_test['ID_pptID']==subj]
        emg_test=emg_test[emg_test['ID_pptID']==subj]
    eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(train_split[0])]
    emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(train_split[0])]

    
    
    sel_cols_emg=args['sel_cols_emg']
    sel_cols_eeg=args['sel_cols_eeg'] 
        
    if args['fusion_alg']['fusion_alg_type']=='svm':      
        if args['get_train_acc']:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train = fusion_SVM(emg_train, eeg_train, emg_test, eeg_test, args)
        else:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_SVM(emg_train, eeg_train, emg_test, eeg_test, args)
    
    elif args['fusion_alg']['fusion_alg_type']=='lda':           
        if args['get_train_acc']:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train = fusion_LDA(emg_train, eeg_train, emg_test, eeg_test, args)
        else:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_LDA(emg_train, eeg_train, emg_test, eeg_test, args)
    
    elif args['fusion_alg']['fusion_alg_type']=='rf':    
        if args['get_train_acc']:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train = fusion_RF(emg_train, eeg_train, emg_test, eeg_test, args)
        else:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_RF(emg_train, eeg_train, emg_test, eeg_test, args)
    
    else:        
        if args['get_train_acc']:
            emg_trainacc=emg_train.copy()
            eeg_trainacc=eeg_train.copy()
            emg_trainacc.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            eeg_trainacc.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
       
        emg_train=ml.drop_ID_cols(emg_train)
        eeg_train=ml.drop_ID_cols(eeg_train)
        
        eeg_train=eeg_train.iloc[:,sel_cols_eeg]
        emg_train=emg_train.iloc[:,sel_cols_emg]
        
        emg_model,eeg_model=train_models_opt(emg_train,eeg_train,args)
    
        classlabels = emg_model.classes_
        
        emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            
        targets, predlist_emg, predlist_eeg, predlist_fusion,_,_,_ = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args, sel_cols_eeg,sel_cols_emg)

        if args['get_train_acc']:
            traintargs, predlist_emgtrain, predlist_eegtrain, predlist_train,_,_,_ = refactor_synced_predict(emg_trainacc, eeg_trainacc, emg_model, eeg_model, classlabels, args, sel_cols_eeg,sel_cols_emg)
    
    gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels=classes_from_preds(targets,predlist_emg,predlist_eeg,predlist_fusion,classlabels)
    '''could calculate log loss if got the probabilities back''' #https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
        
    if args['plot_confmats']:
        gesturelabels=[params.idx_to_gestures[label] for label in classlabels]
        tt.confmat(gest_truth,gest_pred_eeg,gesturelabels,title='EEG')
        tt.confmat(gest_truth,gest_pred_emg,gesturelabels,title='EMG')
        tt.confmat(gest_truth,gest_pred_fusion,gesturelabels,title='Fusion')
        
    emg_acc=(accuracy_score(gest_truth,gest_pred_emg))
    eeg_acc=(accuracy_score(gest_truth,gest_pred_eeg))
    acc=accuracy_score(gest_truth,gest_pred_fusion)

    kappa=(cohen_kappa_score(gest_truth,gest_pred_fusion))
    
    if args['get_train_acc']:
        train_truth=[params.idx_to_gestures[gest] for gest in traintargs]
        train_preds=[params.idx_to_gestures[pred] for pred in predlist_train]
        train_acc=(accuracy_score(train_truth,train_preds))
    else:
        train_acc=(0)

    end=time.time()
    #return 1-mean_acc
    return {
        'loss': 1-acc,
        'status': STATUS_OK,
        'kappa':kappa,
        'fusion_acc':acc,
        'emg_acc':emg_acc,
        'eeg_acc':eeg_acc,
        'train_acc':train_acc,
        'elapsed_time':end-start,}


def fusion_test(emg_train,eeg_train,emg_test,eeg_test,args):
    start=time.time() 
        
    if args['fusion_alg']['fusion_alg_type']=='svm':      
        if args['get_train_acc']:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train = fusion_SVM(emg_train, eeg_train, emg_test, eeg_test, args)
        else:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_SVM(emg_train, eeg_train, emg_test, eeg_test, args)
    
    elif args['fusion_alg']['fusion_alg_type']=='lda':           
        if args['get_train_acc']:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train = fusion_LDA(emg_train, eeg_train, emg_test, eeg_test, args)
        else:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_LDA(emg_train, eeg_train, emg_test, eeg_test, args)
    
    elif args['fusion_alg']['fusion_alg_type']=='rf':    
        if args['get_train_acc']:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train = fusion_RF(emg_train, eeg_train, emg_test, eeg_test, args)
        else:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_RF(emg_train, eeg_train, emg_test, eeg_test, args)
    
    else:        
        if args['get_train_acc']:
            emg_trainacc=emg_train.copy()
            eeg_trainacc=eeg_train.copy()
            emg_trainacc.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            eeg_trainacc.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
       
        emg_train=ml.drop_ID_cols(emg_train)
        eeg_train=ml.drop_ID_cols(eeg_train)
        
        eeg_train=eeg_train.iloc[:,args['sel_cols_eeg']]
        emg_train=emg_train.iloc[:,args['sel_cols_emg']]
        
        emg_model,eeg_model=train_models_opt(emg_train,eeg_train,args)
    
        classlabels = emg_model.classes_
        
        emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            
        targets, predlist_emg, predlist_eeg, predlist_fusion,_,_,_ = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args, sel_cols_eeg,sel_cols_emg)

        if args['get_train_acc']:
            traintargs, predlist_emgtrain, predlist_eegtrain, predlist_train,_,_,_ = refactor_synced_predict(emg_trainacc, eeg_trainacc, emg_model, eeg_model, classlabels, args, sel_cols_eeg,sel_cols_emg)
    
    gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels=classes_from_preds(targets,predlist_emg,predlist_eeg,predlist_fusion,classlabels)
    '''could calculate log loss if got the probabilities back''' #https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
        
    if args['plot_confmats']:
        gesturelabels=[params.idx_to_gestures[label] for label in classlabels]
        tt.confmat(gest_truth,gest_pred_eeg,gesturelabels,title=('Ppt '+args['subject id']+' EEG'))
        tt.confmat(gest_truth,gest_pred_emg,gesturelabels,title=('Ppt '+args['subject id']+' EMG'))
        tt.confmat(gest_truth,gest_pred_fusion,gesturelabels,title=('Ppt '+args['subject id']+' Fusion'))
        
    emg_acc=(accuracy_score(gest_truth,gest_pred_emg))
    eeg_acc=(accuracy_score(gest_truth,gest_pred_eeg))
    acc=accuracy_score(gest_truth,gest_pred_fusion)

    kappa=(cohen_kappa_score(gest_truth,gest_pred_fusion))
    
    if args['get_train_acc']:
        train_truth=[params.idx_to_gestures[gest] for gest in traintargs]
        train_preds=[params.idx_to_gestures[pred] for pred in predlist_train]
        train_acc=(accuracy_score(train_truth,train_preds))
    else:
        train_acc=(0)

    end=time.time()
    #return 1-mean_acc
    return {
        'loss': 1-acc,
        'status': STATUS_OK,
        'kappa':kappa,
        'fusion_acc':acc,
        'emg_acc':emg_acc,
        'eeg_acc':eeg_acc,
        'train_acc':train_acc,
        'elapsed_time':end-start,}

def scale_nonSubj(emg_others,eeg_others,augment_scale):
    emg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_others=emg_others.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_others=eeg_others.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    
    eeg_others['ID_stratID'] = eeg_others['ID_run'].astype(str)+eeg_others['ID_pptID'].astype(str)+eeg_others['Label'].astype(str)+eeg_others['ID_gestrep'].astype(str)
    emg_others['ID_stratID'] = emg_others['ID_run'].astype(str)+emg_others['ID_pptID'].astype(str)+emg_others['Label'].astype(str)+emg_others['ID_gestrep'].astype(str)
    random_split=random.randint(0,100)
    
    if not emg_others['ID_stratID'].equals(eeg_others['ID_stratID']):
        raise ValueError('EMG & EEG performances misaligned')
    gest_perfs=emg_others['ID_stratID'].unique()
    gest_strat=pd.DataFrame([gest_perfs,[(perf.split('.')[1])+(perf.split('.')[2][-1]) for perf in gest_perfs]]).transpose()
    
    _,augment_split=train_test_split(gest_strat,test_size=augment_scale,
                                          random_state=random_split,stratify=gest_strat[1])
    
    emg_aug=emg_others[emg_others['ID_stratID'].isin(augment_split[0])]
    eeg_aug=eeg_others[eeg_others['ID_stratID'].isin(augment_split[0])]
    
    return emg_aug,eeg_aug





if __name__ == '__main__':
    
    run_test=False
    plot_results=True
    load_res_path=None
    load_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ2\D1_AugAllfinal_resMinimal.csv"
  #  load_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ2\B3_AugTrain_rolloff1.0_augment0.167_resMinimal.csv"
    load_res_path=r"/home/michael/Downloads/D1_AugAllfinal_resMinimal.csv"

    systemUnderTest = 'D1_AugAll'
    rolling_off_subj=True
    
    testset_size = 0.33
    
        
    if systemUnderTest == 'A1_FullBespoke':

        feats_method='subject'
        train_method='subject'
        augment_scales=[0]
        
        if rolling_off_subj==True:
            '''
            train_sizes1=1.099-np.geomspace(0.1,0.2,5)[::-1]
            train_sizes_mid=np.linspace(0.2925,0.8,4)
            train_sizes2=np.geomspace(0.02,0.2,5)
            train_sizes3=np.geomspace(0.01,0.02,5)
            train_sizes=np.unique(np.concatenate((train_sizes1,train_sizes_mid,train_sizes2,train_sizes3)))
            '''
            train_sizes=np.unique(np.concatenate((np.linspace(0.01,0.1,3),np.geomspace(0.1,0.6,6),np.geomspace(0.6,1,4))))
        else:
            train_sizes=[1]
        
    elif systemUnderTest == 'D1_AugAll':
       # train_sizes=[0.25]
       # train_sizes=np.geomspace(0.01,1,5)
     #  train_sizes=np.linspace(0.01,1,5)
        #manually added 0.05 and 0.1 as 0.01 was too small
        
        train_sizes=np.concatenate(([0.05,0.1],np.linspace(0.01,1,5)[1:]))
        
        feats_method='non-subject aug'
        opt_method='non-subject aug'
        train_method='non-subject aug'
        #augment_scales = np.geomspace(0.02,0.33,4)
        # 0.00666 would be 1 full gesture per person, for a set of 19
        # ie 1/150, because each gesture was done 50 times on 3 days = 150 per gest per ppt
        # below coerces them to be multiples of 0.00666 ie to ensure equal # per ppt per class
        augment_scales=[0,0.00666,0.02,0.05263,0.166]#,0.33,0.67]
        # the scales above are 0, 1, 3, not 6, 7.89, not 12 (0.08), 25, 50, 100 per ppt per class
        # 0.05263 is 1/19, 7.89 per gest per ppt, i.e. result in aug_size = train_size
            #(actually ends up as 0.05333 = 8 per class per ppt = 152 in the aug)
        # 100 per class per ppt is the same amount as left over in the training set after 0.33 reserved for test
        # 50 and 100 removed for now for practicality as very big! dwarfs the subject
        augment_scales=[0.1, 0.075, 0.33, 0.45, 0.25, 0.67]
        augment_scales = np.array([round(scale/(1/150))*(1/150) for scale in augment_scales])
    
    if run_test:
        iters = 100
        
        emg_set_path=params.jeong_emg_noholdout
        eeg_set_path=params.jeong_eeg_noholdout
        
        emg_set=ml.pd.read_csv(emg_set_path,delimiter=',')
        eeg_set=ml.pd.read_csv(eeg_set_path,delimiter=',')
        emg_set,eeg_set=balance_set(emg_set,eeg_set)
        #space.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True})
    
        eeg_masks=get_ppt_split_flexi(eeg_set)
        emg_masks=get_ppt_split_flexi(emg_set)
        
        ppt_winners=[]
        ppt_results=[]
        skipRolloff=False
        
        for rolloff in train_sizes:
            for augment_scale in augment_scales:
                for idx,emg_mask in enumerate(emg_masks):
                    space=setup_search_space(architecture='decision',include_svm=True)
                    
                    space.update({'l1_sparsity':0.05})
                    space.update({'l1_maxfeats':40})
                    
                    space.update({'rolloff_factor':rolloff})
                    space.update({'augment_scale':augment_scale})
                    
                    trials=Trials()
                    
                    space.update({'testset_size':testset_size,})
                    
                    eeg_mask=eeg_masks[idx]
                    
                    emg_ppt = emg_set[emg_mask]
                    eeg_ppt = eeg_set[eeg_mask]
                    
                    
                    emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
                    eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
                    
                    index_emg=ml.pd.MultiIndex.from_arrays([emg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
                    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
                    emg_ppt=emg_ppt.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
                    eeg_ppt=eeg_ppt.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
                    
                    eeg_ppt['ID_stratID']=eeg_ppt['ID_run'].astype(str)+eeg_ppt['Label'].astype(str)+eeg_ppt['ID_gestrep'].astype(str)
                    emg_ppt['ID_stratID']=emg_ppt['ID_run'].astype(str)+eeg_ppt['Label'].astype(str)+eeg_ppt['ID_gestrep'].astype(str)
                    random_split=random.randint(0,100)
                    
                    if not emg_ppt['ID_stratID'].equals(eeg_ppt['ID_stratID']):
                        raise ValueError('EMG & EEG performances misaligned')
                    gest_perfs=emg_ppt['ID_stratID'].unique()
                    gest_strat=pd.DataFrame([gest_perfs,[perf.split('.')[1][-1] for perf in gest_perfs]]).transpose()
                    
                    remainder,test_split=train_test_split(gest_strat,test_size=testset_size,
                                                          random_state=random_split,stratify=gest_strat[1])
                    
                    if space['rolloff_factor'] < 1:
                        remainder,_=train_test_split(remainder,train_size=space['rolloff_factor'],random_state=random_split,stratify=remainder[1])
                        if min(remainder[1].value_counts()) < 2:
                            print('rolloff of ' +str(space['rolloff_factor'])+' results in < 2 performances per class')
                            skipRolloff=True
                            break
        
                    
                    eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split[0])]
                    emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]
                    eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(remainder[0])]
                    emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(remainder[0])]
                                 
                         
                    emg_train,emgscaler=feats.scale_feats_train(emg_train,space['scalingtype'])
                    eeg_train,eegscaler=feats.scale_feats_train(eeg_train,space['scalingtype'])
                    emg_test=feats.scale_feats_test(emg_test,emgscaler)
                    eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
                                    
                    
                    emg_others = emg_set[~emg_mask]
                    eeg_others = eeg_set[~eeg_mask]
                    
                    if augment_scale == 0:
                        emg_joint = emg_train
                        eeg_joint = eeg_train
                    else:
                        emg_aug,eeg_aug = scale_nonSubj(emg_others,eeg_others,augment_scale)

                        emg_aug=feats.scale_feats_test(emg_aug,emgscaler)
                        eeg_aug=feats.scale_feats_test(eeg_aug,eegscaler)
                        
                        emg_joint = pd.concat([emg_train,emg_aug])
                        eeg_joint = pd.concat([eeg_train,eeg_aug])


                    if feats_method=='subject':
                        sel_cols_emg=feats.sel_percent_feats_df(ml.drop_ID_cols(emg_train),percent=15)
                        sel_cols_emg=np.append(sel_cols_emg,ml.drop_ID_cols(emg_train).columns.get_loc('Label'))
                        sel_cols_eeg=feats.sel_feats_l1_df(ml.drop_ID_cols(eeg_train),sparsityC=space['l1_sparsity'],maxfeats=space['l1_maxfeats'])
                        sel_cols_eeg=np.append(sel_cols_eeg,ml.drop_ID_cols(eeg_train).columns.get_loc('Label')) 
                    
                    elif feats_method=='non-subject aug':                     
                        sel_cols_emg=feats.sel_percent_feats_df(ml.drop_ID_cols(emg_joint),percent=15)
                        sel_cols_emg=np.append(sel_cols_emg,ml.drop_ID_cols(emg_joint).columns.get_loc('Label'))
                        sel_cols_eeg=feats.sel_feats_l1_df(ml.drop_ID_cols(eeg_joint),sparsityC=space['l1_sparsity'],maxfeats=space['l1_maxfeats'])
                        sel_cols_eeg=np.append(sel_cols_eeg,ml.drop_ID_cols(eeg_joint).columns.get_loc('Label')) 
                    
 
                    space['sel_cols_emg']=sel_cols_emg
                    space['sel_cols_eeg']=sel_cols_eeg
                    space['subject-id']=eeg_ppt['ID_pptID'][0]                    
                    
                    if opt_method=='subject':
                        space.update({'emg_set':emg_train,'eeg_set':eeg_train,'data_in_memory':True,'prebalanced':True})
                    elif opt_method=='non-subject aug':
                        space.update({'emg_set':emg_joint,'eeg_set':eeg_joint,'data_in_memory':True,'prebalanced':True})
                    
                    space.update({'featsel_method':feats_method})
                    space.update({'train_method':train_method})
                    space.update({'opt_method':opt_method})
                    
                    best = fmin(fuse_fullbespoke,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=iters,
                            trials=trials)
                    
                    winner_args=space_eval(space,best)
                    best_loss=trials.best_trial['result']['loss']
            
                    winner_args['sel_cols_emg']=sel_cols_emg
                    winner_args['sel_cols_eeg']=sel_cols_eeg 
                    winner_args['plot_confmats']=True
                    winner_args['subject id']=str(int(eeg_ppt['ID_pptID'][0]))
                    
                    subject_results=fusion_test(emg_joint,eeg_joint,emg_test,eeg_test,winner_args)
                    subject_results['best_loss']=best_loss
                    
                    ppt_winners.append(winner_args)
                    ppt_results.append(subject_results)
    
                if skipRolloff:
                    skipRolloff=False
                    continue
                else:
                    skipRolloff=False
                results_final=pd.DataFrame(ppt_results)
                winners_final=pd.DataFrame(ppt_winners)
                winners_final=winners_final.drop(['bag_eeg','data_in_memory','eeg_set','emg_set','prebalanced','testset_size',
                                                  'eeg_set_path','emg_set_path','using_literature_data','stack_distros',
                                                  'scalingtype','plot_confmats','l1_maxfeats','get_train_acc',],axis=1)    
                #winners_final=pd.concat([winners_final.eeg.apply(pd.Series), winners_final.drop('eeg', axis=1)], axis=1)
                #winners_final=pd.concat([winners_final.emg.apply(pd.Series), winners_final.drop('emg', axis=1)], axis=1)
                #winners_final=pd.concat([winners_final.fusion_alg.apply(pd.Series), winners_final.drop('fusion_alg', axis=1)], axis=1)
                    
                for col in ['emg','eeg','fusion_alg']:
                    winners_final=pd.concat([winners_final.drop(col,axis=1),
                                             winners_final[col].apply(lambda x: pd.Series(x)).rename(columns=lambda x: f'{col}_{x}')],axis=1)    
                    
                results=results_final.join(winners_final)
                results['opt_acc']=1-results['best_loss']
                scores_minimal=results[['subject id','fusion_acc','emg_acc','eeg_acc','elapsed_time',
                               'fusion_alg_fusion_alg_type','eeg_eeg_model_type','emg_emg_model_type',
                               'featsel_method','train_method','rolloff_factor','augment_scale','best_loss','opt_acc']]
                
                currentpath=os.path.dirname(__file__)
                result_dir=params.jeong_results_dir
                resultpath=os.path.join(currentpath,result_dir)    
                resultpath=os.path.join(resultpath,'RQ2')
               
                if type(train_sizes)==type([1]):
                    if rolloff==1:
                        rolloff_label=''
                    else:
                        rolloff_label='_rolloff'+str(round(rolloff,3))
                else:
                    rolloff_label='_rolloff'+str(round(rolloff,3))
                
                if augment_scale==0:
                    augment_label=''
                else:
                    augment_label='_augment'+str(round(augment_scale,3))
                
                picklepath=os.path.join(resultpath,(systemUnderTest+rolloff_label+augment_label+'_resDF.pkl'))
                csvpath=os.path.join(resultpath,(systemUnderTest+rolloff_label+augment_label+'_resMinimal.csv'))
        
                pickle.dump(results,open(picklepath,'wb'))
                scores_minimal.to_csv(csvpath)
        
        picklefullpath=os.path.join(resultpath,(systemUnderTest+'final_resDF.pkl'))
        csvfullpath=os.path.join(resultpath,(systemUnderTest+'final_resMinimal.csv'))

        pickle.dump(results,open(picklefullpath,'wb'))
        scores_minimal.to_csv(csvfullpath)
    else:
        scores_minimal=pd.read_csv(load_res_path,index_col=0)        
    
    if plot_results:    
        fig,ax=plt.subplots();
        for ppt in scores_minimal['subject id'].unique():
            scores_minimal[scores_minimal['subject id']==ppt].plot(y='fusion_acc',x='rolloff_factor',ax=ax,color='tab:blue',legend=None)
            scores_minimal[scores_minimal['subject id']==ppt].plot(y='opt_acc',x='rolloff_factor',ax=ax,color='tab:orange',legend=None)
            
        #    scores_minimal[scores_minimal['subject id']==ppt].plot(y='emg_acc',x='rolloff_factor',ax=ax,color='tab:green',legend=None)
        #    scores_minimal[scores_minimal['subject id']==ppt].plot(y='eeg_acc',x='rolloff_factor',ax=ax,color='tab:purple',legend=None)
        #ax.set_xlim(0.05,0.4)
            
        fig,ax=plt.subplots();
        for ppt in scores_minimal['subject id'].unique():
            scores_minimal[scores_minimal['subject id']==ppt].plot(y='fusion_acc',x='augment_scale',ax=ax,color='tab:green',legend=None)
       #     scores_minimal[scores_minimal['subject id']==ppt].plot(y='opt_acc',x='augment_scale',ax=ax,color='tab:purple',legend=None)
        ax.set_ylim(0.25,1)
        
        fig,ax=plt.subplots();
        ax.scatter(scores_minimal['rolloff_factor'],scores_minimal['augment_scale'],c=scores_minimal['fusion_acc'],cmap='copper')
        
        for ppt in scores_minimal['subject id'].unique():
            fig,ax=plt.subplots();
         #   scores_minimal[scores_minimal['subject id']==ppt].plot.scatter(y='augment_scale',x='rolloff_factor',c='fusion_acc',
         #                                                                  ax=ax,cmap='copper')
            
           # '''
            plt.rcParams['figure.dpi'] = 150 # DEFAULT IS 100
            subj= scores_minimal[scores_minimal['subject id']==ppt]
            fullbesp=subj[subj['rolloff_factor']==1][subj['augment_scale']==0]['fusion_acc'].item()
         #   subj['relative_to_bespoke']=subj['fusion_acc']-fullbesp
         #   plt.scatter(subj['rolloff_factor'],subj['augment_scale'],c=subj['relative_to_bespoke'])
            plt.scatter(subj['rolloff_factor'],subj['augment_scale'],c=subj['fusion_acc'],norm=PowerNorm(np.e))
            plt.xlabel('Proportion of subject\'s 67% non-test data')
            #plt.yticks(rotation=33)
            plt.ylabel('Proportion of non-subj augmenting Everything')
            plt.colorbar()
            plt.title('Subject '+str(ppt)+'. No rolloff, no aug = '+str(round(fullbesp,5)))
            wincoords=tuple(subj.loc[subj['fusion_acc'].idxmax()][['rolloff_factor','augment_scale']].tolist())
            plt.annotate(str(round(subj['fusion_acc'].max(),3)),wincoords)
            
            tolerance=0.0001 # 0.01% is close enough
            meetsOrBeats=subj.loc[subj['fusion_acc']>fullbesp-tolerance][['rolloff_factor','augment_scale']]
            for _,row in meetsOrBeats.iterrows():
                #ax.add_patch(plt.Circle((row['rolloff_factor'],row['augment_scale']),0.07,color='r',fill=False))
                ax.add_patch(Ellipse((row['rolloff_factor'],row['augment_scale']),width=0.05,height=0.01,color='r',fill=False))
            
            plt.show()
         #   '''
            
        plot_all_rollofs=True    
        for ppt in scores_minimal['subject id'].unique():
            fig,ax=plt.subplots();

            plt.rcParams['figure.dpi'] = 150 # DEFAULT IS 100
            subj= scores_minimal[scores_minimal['subject id']==ppt]
            fullbesp=subj[subj['rolloff_factor']==1][subj['augment_scale']==0]['fusion_acc'].item()
            
            if plot_all_rollofs:
                for rolloff in np.sort(subj['rolloff_factor'].unique()):
                    subj[subj['rolloff_factor']==rolloff].plot(x='augment_scale',y='fusion_acc',marker='.',ax=ax)
     #           ax.set_ylim((0.65,0.95))
            else:
                subj=subj[subj['rolloff_factor']>0.15]
                for rolloff in np.sort(subj['rolloff_factor'].unique()):
                    subj[subj['rolloff_factor']==rolloff].plot(x='augment_scale',y='fusion_acc',marker='.',ax=ax)
                
            ax.legend(np.sort(subj['rolloff_factor'].unique()),title='Proportion subject data')
            
            plt.title('Subject '+str(ppt)+'. No rolloff, no aug = '+str(round(fullbesp,5)))
            
            wincoords=tuple(subj.loc[subj['fusion_acc'].idxmax()][['augment_scale','fusion_acc']].tolist())
            plt.annotate(str(round(subj['fusion_acc'].max(),3)),wincoords)
            
       #     tolerance=0.0001 # 0.01% is close enough
       #     meetsOrBeats=subj.loc[subj['fusion_acc']>fullbesp-tolerance][['rolloff_factor','augment_scale']]
            # np.isclose ??
       #     for _,row in meetsOrBeats.iterrows():
                #ax.add_patch(plt.Circle((row['rolloff_factor'],row['augment_scale']),0.07,color='r',fill=False))
       #         ax.add_patch(Ellipse((row['rolloff_factor'],row['augment_scale']),width=0.05,height=0.01,color='r',fill=False))
            
            #ax.set_xticks(subj['augment_scale'].unique())
 #           ax.set_xlabel('Proportion of non-subj augmenting Everything')
            
            plt.hlines(y=0.75,label='Generalist',xmin=0,xmax=0.15,linestyles='--')
            plt.show()
            
        print('*****\nHeavily affected by randomness, BUT I think it may be in part the',
              'randomness of which bits of non-subj are chosen to be added. IE not randomness in model etc causing',
              'effect where there is none, but rather randomness as to how effective it will be dependent on',
              'the non-subj data that is most helpful (or most helpful to this subj) which one could',
              'theoretically identify statically or find way to auto identify.\n*****')
    
    #chance we confuse it eg that by adding non subject within opt, all its doing is learning to firstly
        #ignore the bits of training data that are non-subject, then learn helpful things from the subject data
    bespokescores={}
    for ppt in scores_minimal['subject id'].unique():

        subj= scores_minimal[scores_minimal['subject id']==ppt]
        fullbesp=subj[subj['rolloff_factor']==1][subj['augment_scale']==0]['fusion_acc'].item()
        bespokescores.update({ppt:fullbesp})
    
    scores_minimal['bespokescore']=scores_minimal['subject id'].map(bespokescores)


    scores_minimal['change']=scores_minimal['fusion_acc']-scores_minimal['bespokescore']
    
    fig,ax=plt.subplots();
    for ppt in scores_minimal['subject id'].unique():
        subj= scores_minimal[scores_minimal['subject id']==ppt]
        for rolloff in np.sort(subj['rolloff_factor'].unique()):
            subj[subj['rolloff_factor']==rolloff].plot(x='augment_scale',y='change',marker='.',ax=ax,legend=None)
    
    fig,ax=plt.subplots();
    for rolloffLevel in np.sort(scores_minimal['rolloff_factor'].unique()):
        rolloff=scores_minimal[scores_minimal['rolloff_factor']==rolloffLevel]
        rolloffscores={}
        for auglevel in np.sort(rolloff['augment_scale'].unique()):
            aug_avg=np.average(rolloff[rolloff['augment_scale']==auglevel]['fusion_acc'])
            rolloffscores.update({auglevel:aug_avg})
        pd.DataFrame(rolloffscores.items(),columns=['Aug level','Accuracy']).plot(x='Aug level',y='Accuracy',ax=ax,marker='.')
    ax.legend(np.sort(scores_minimal['rolloff_factor'].unique()),title='Proportion subject data')

    '''
    fig,ax=plt.subplots();
    for rolloffLevel in np.sort(scores_minimal['rolloff_factor'].unique()):
        rolloff=scores_minimal[scores_minimal['rolloff_factor']==rolloffLevel]
        rolloffscores={}
        rolloffstds={}
        for auglevel in np.sort(rolloff['augment_scale'].unique()):
            aug_avg=np.average(rolloff[rolloff['augment_scale']==auglevel]['fusion_acc'])
            aug_std=np.std(rolloff[rolloff['augment_scale']==auglevel]['fusion_acc'])
            rolloffscores.update({auglevel:aug_avg})
            rolloffstds.update({auglevel:aug_std})
        pd.DataFrame(rolloffscores.items(),columns=['Aug level','Accuracy']).plot(x='Aug level',y='Accuracy',ax=ax,marker='.')
    ax.legend(np.sort(scores_minimal['rolloff_factor'].unique()),title='Proportion subject data')
    '''
    fig,ax=plt.subplots();
    for key,group in scores_minimal.groupby('rolloff_factor'):
        grouped=group.groupby(['augment_scale'])['fusion_acc'].agg(['mean','std']).reset_index()
        plt.errorbar(x=grouped['augment_scale'],y=grouped['mean'],yerr=grouped['std'],marker='.',capsize=5)
    ax.set_ylim(0.6,0.95)
    plt.show()
    
    fig,ax=plt.subplots();
    for key,group in scores_minimal.groupby('rolloff_factor'):
        grouped=group.groupby(['augment_scale'])['change'].agg(['mean','std']).reset_index()
        plt.errorbar(x=grouped['augment_scale'],y=grouped['mean'],yerr=grouped['std'],marker='.',capsize=5)
    plt.show()
    

