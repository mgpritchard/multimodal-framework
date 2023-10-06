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
    '''    
    emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_ppt=emg_ppt.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_ppt=eeg_ppt.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    
    eeg_ppt['ID_stratID']=eeg_ppt['ID_run'].astype(str)+eeg_ppt['Label'].astype(str)+eeg_ppt['ID_gestrep'].astype(str)
    emg_ppt['ID_stratID']=emg_ppt['ID_run'].astype(str)+eeg_ppt['Label'].astype(str)+eeg_ppt['ID_gestrep'].astype(str)
    '''
    
    if not emg_ppt['ID_stratID'].equals(eeg_ppt['ID_stratID']):
        raise ValueError('EMG & EEG performances misaligned')
    gest_perfs=emg_ppt['ID_stratID'].unique()
    gest_strat=pd.DataFrame([gest_perfs,[perf.split('.')[1][-1] for perf in gest_perfs]]).transpose()
    
    random_split=random.randint(0,100)
    if args['featsel_method']=='non-subject':
        remainder,test_split=train_test_split(gest_strat,test_size=args['testset_size'],
                                                random_state=random_split,stratify=gest_strat[1])
                
        eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split[0])]
        emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]
        eeg_remainder=eeg_ppt[eeg_ppt['ID_stratID'].isin(remainder[0])]
        emg_remainder=emg_ppt[emg_ppt['ID_stratID'].isin(remainder[0])]

        train_split,discard=train_test_split(remainder,test_size=args['opt_size'],
                                                 random_state=random_split,stratify=remainder[1])
        
        eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(train_split[0])]
        emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(train_split[0])]
        
        emg_featsel=None
        eeg_featsel=None
    elif args['featsel_method']=='subject':
        remainder,test_split=train_test_split(gest_strat,test_size=args['testset_size'],
                                                random_state=random_split,stratify=gest_strat[1])
        
        eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split[0])]
        emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]
        eeg_remainder=eeg_ppt[eeg_ppt['ID_stratID'].isin(remainder[0])]
        emg_remainder=emg_ppt[emg_ppt['ID_stratID'].isin(remainder[0])]
                
        remainder,featsel_split=train_test_split(remainder,test_size=args['featsel_size'],
                                                 random_state=random_split,stratify=remainder[1])
        
        eeg_featsel=eeg_ppt[eeg_ppt['ID_stratID'].isin(featsel_split[0])]
        emg_featsel=emg_ppt[emg_ppt['ID_stratID'].isin(featsel_split[0])]
        eeg_remainder=eeg_ppt[eeg_ppt['ID_stratID'].isin(remainder[0])]
        emg_remainder=emg_ppt[emg_ppt['ID_stratID'].isin(remainder[0])]

        train_split,discard=train_test_split(remainder,test_size=args['opt_size']/(1-args['featsel_size']),
                                                 random_state=random_split,stratify=remainder[1])
        
        eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(train_split[0])]
        emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(train_split[0])]


    ''' it would feel more intuitive to scale by the feat_selection split, 
    but then if feat selection is based on NON-subject data, we'd have to use
    non-subject data for scaling, which seems a needless setback if we can avoid'''
    emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
    eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
    emg_test=feats.scale_feats_test(emg_test,emgscaler)
    eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
    
    if args['featsel_method']=='subject':
        emg_featsel=feats.scale_feats_test(emg_featsel,emgscaler)
        eeg_featsel=feats.scale_feats_test(eeg_featsel,eegscaler)
    
        sel_cols_emg=feats.sel_percent_feats_df(ml.drop_ID_cols(emg_featsel),percent=15)
        sel_cols_emg=np.append(sel_cols_emg,ml.drop_ID_cols(emg_featsel).columns.get_loc('Label'))
        sel_cols_eeg=feats.sel_feats_l1_df(ml.drop_ID_cols(eeg_featsel),sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
        sel_cols_eeg=np.append(sel_cols_eeg,ml.drop_ID_cols(eeg_featsel).columns.get_loc('Label'))
    
    elif args ['featsel_method']=='non-subject':
        idx=int(eeg_test['ID_pptID'].iloc[0])-1
        sel_cols_eeg=[ml.drop_ID_cols(eeg_train).columns.get_loc(col) for col in args['eeg_feats_nonsubj'].iloc[idx].tolist()]
        sel_cols_emg=[ml.drop_ID_cols(emg_train).columns.get_loc(col) for col in args['emg_feats_nonsubj'].iloc[idx].tolist()]
    
    args['sel_cols_emg']=sel_cols_emg
    args['sel_cols_eeg']=sel_cols_eeg    
        
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
    
#    elif args['fusion_alg']['fusion_alg_type']=='hierarchical':                           
#                    
#        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_hierarchical(emg_train, eeg_train, emg_test, eeg_test, args)
#
#    elif args['fusion_alg']=='just_emg':      
#        if not args['get_train_acc']:
#            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=only_EMG(emg_train, eeg_train, emg_test, eeg_test, args)
#        else:
#            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=only_EMG(emg_train, eeg_train, emg_test, eeg_test, args)
    
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
    
#    elif args['fusion_alg']['fusion_alg_type']=='hierarchical':                           
#                    
#        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_hierarchical(emg_train, eeg_train, emg_test, eeg_test, args)
#
#    elif args['fusion_alg']=='just_emg':      
#        if not args['get_train_acc']:
#            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=only_EMG(emg_train, eeg_train, emg_test, eeg_test, args)
#        else:
#            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=only_EMG(emg_train, eeg_train, emg_test, eeg_test, args)
    
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




if __name__ == '__main__':

    systemUnderTest = 'B1_FullBespoke_NonSubjFeatSel'
    
    testset_size = 0.33
    
    if systemUnderTest == 'A1_FullBespoke':
        
        featsel_size = 0.25 #intuitively 0.1, but we divide evenly
        train_size = 0.25 #intuitively 0.33
        opt_size = 0.5 #intuitively 0.24. its 2/N, for model sel AND hyperp tune
        #these splits are all OF THE NON-TEST DATA. *0.67 for ratio of the WHOLE data
        
        feats_method='subject'
    elif systemUnderTest == 'B1_FullBespoke_NonSubjFeatSel':
        
        feats_method='non-subject'
        
        featsel_size = 0.0 #intuitively 0.1, but we divide evenly
        train_size = 0.5 #intuitively 0.33
        opt_size = 0.5
    
        eeg_feats_nonsubj=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        emg_feats_nonsubj=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
    
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
    
    for idx,emg_mask in enumerate(emg_masks):
        space=setup_search_space(architecture='decision',include_svm=True)
        space.update({'l1_sparsity':0.05})
        space.update({'l1_maxfeats':40})
        
        trials=Trials()
        
        space.update({'testset_size':testset_size,
                     'featsel_size':featsel_size,
                     'train_size':train_size,
                     'opt_size':opt_size})
        
        eeg_mask=eeg_masks[idx]
        
        emg_ppt = emg_set[emg_mask]
        #emg_others = emg_set[~emg_mask]
        eeg_ppt = eeg_set[eeg_mask]
        #eeg_others = eeg_set[~eeg_mask]
        
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
        
        eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split[0])]
        emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]
        eeg_remainder=eeg_ppt[eeg_ppt['ID_stratID'].isin(remainder[0])]
        emg_remainder=emg_ppt[emg_ppt['ID_stratID'].isin(remainder[0])]


        if feats_method=='subject':
            
            remainder,featsel_split=train_test_split(remainder,test_size=featsel_size,
                                                     random_state=random_split,stratify=remainder[1])
            
            eeg_featsel=eeg_ppt[eeg_ppt['ID_stratID'].isin(featsel_split[0])]
            emg_featsel=emg_ppt[emg_ppt['ID_stratID'].isin(featsel_split[0])]
            eeg_remainder=eeg_ppt[eeg_ppt['ID_stratID'].isin(remainder[0])]
            emg_remainder=emg_ppt[emg_ppt['ID_stratID'].isin(remainder[0])]
        
        elif feats_method=='non-subject':
            space.update({'eeg_feats_nonsubj':eeg_feats_nonsubj,
                          'emg_feats_nonsubj':emg_feats_nonsubj})


        train_split,opt_split=train_test_split(remainder,test_size=opt_size/(1-featsel_size),
                                                 random_state=random_split,stratify=remainder[1])
        
        eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(train_split[0])]
        emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(train_split[0])]

        eeg_opt=eeg_ppt[eeg_ppt['ID_stratID'].isin(opt_split[0])]
        emg_opt=emg_ppt[emg_ppt['ID_stratID'].isin(opt_split[0])]
                   
        
        space.update({'emg_set':emg_opt,'eeg_set':eeg_opt,'data_in_memory':True,'prebalanced':True})
        
        space.update({'featsel_method':feats_method})
        
        best = fmin(fuse_fullbespoke,
                space=space,
                algo=tpe.suggest,
                max_evals=iters,
                trials=trials)
        
        winner_args=space_eval(space,best)
        
        emg_train,emgscaler=feats.scale_feats_train(emg_train,winner_args['scalingtype'])
        eeg_train,eegscaler=feats.scale_feats_train(eeg_train,winner_args['scalingtype'])
        emg_test=feats.scale_feats_test(emg_test,emgscaler)
        eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
        
        if winner_args['featsel_method']=='subject':
            if featsel_size > 0:
                emg_featsel=feats.scale_feats_test(emg_featsel,emgscaler)
                eeg_featsel=feats.scale_feats_test(eeg_featsel,eegscaler)
            
                sel_cols_emg=feats.sel_percent_feats_df(ml.drop_ID_cols(emg_featsel),percent=15)
                sel_cols_emg=np.append(sel_cols_emg,ml.drop_ID_cols(emg_featsel).columns.get_loc('Label'))
                sel_cols_eeg=feats.sel_feats_l1_df(ml.drop_ID_cols(eeg_featsel),sparsityC=winner_args['l1_sparsity'],maxfeats=winner_args['l1_maxfeats'])
                sel_cols_eeg=np.append(sel_cols_eeg,ml.drop_ID_cols(eeg_featsel).columns.get_loc('Label'))
            else:
                sel_cols_emg=feats.sel_percent_feats_df(ml.drop_ID_cols(emg_train),percent=15)
                sel_cols_emg=np.append(sel_cols_emg,ml.drop_ID_cols(emg_train).columns.get_loc('Label'))
                sel_cols_eeg=feats.sel_feats_l1_df(ml.drop_ID_cols(eeg_train),sparsityC=winner_args['l1_sparsity'],maxfeats=winner_args['l1_maxfeats'])
                sel_cols_eeg=np.append(sel_cols_eeg,ml.drop_ID_cols(eeg_train).columns.get_loc('Label'))
        
        elif winner_args['featsel_method']=='non-subject':
            idx=int(eeg_test['ID_pptID'].iloc[0])-1
            sel_cols_eeg=[ml.drop_ID_cols(eeg_train).columns.get_loc(col) for col in winner_args['eeg_feats_nonsubj'].iloc[idx].tolist()]
            sel_cols_emg=[ml.drop_ID_cols(emg_train).columns.get_loc(col) for col in winner_args['emg_feats_nonsubj'].iloc[idx].tolist()]
        
        winner_args['sel_cols_emg']=sel_cols_emg
        winner_args['sel_cols_eeg']=sel_cols_eeg 
        winner_args['plot_confmats']=True
        winner_args['subject id']=str(int(eeg_ppt['ID_pptID'][0]))
        
        subject_results=fusion_test(emg_train,eeg_train,emg_test,eeg_test,winner_args)
        
        ppt_winners.append(winner_args)
        ppt_results.append(subject_results)
        
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
    
    currentpath=os.path.dirname(__file__)
    result_dir=params.jeong_results_dir
    resultpath=os.path.join(currentpath,result_dir)    
    resultpath=os.path.join(resultpath,'RQ2')
    picklepath=os.path.join(resultpath,(systemUnderTest+'_resDF.pkl'))
    pickle.dump(results,open(picklepath,'wb'))
    
    scores_minimal=results[['subject id','fusion_acc','emg_acc','eeg_acc','elapsed_time',
                   'fusion_alg_fusion_alg_type','eeg_eeg_model_type','emg_emg_model_type',
                   'featsel_method','featsel_size','opt_size','train_size']]
    csvpath=os.path.join(resultpath,(systemUnderTest+'_resMinimal.csv'))
    scores_minimal.to_csv(csvpath)




    