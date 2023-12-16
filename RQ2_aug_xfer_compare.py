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
from functools import partial
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


gen_dev_accs={2: 0.76625, 3: 0.68875, 4: 0.7879166666666667, 5: 0.77875, 7: 0.81, 8: 0.745, 9: 0.6504166666666666,
              10: 0.6991666666666667, 12: 0.70375, 13: 0.5275, 14: 0.6008333333333333, 15: 0.65875,
              17: 0.8183333333333334, 18: 0.74875, 19: 0.7379166666666667, 20: 0.7408333333333333,
              22: 0.7375, 23: 0.6825, 24: 0.8375, 25: 0.7395833333333334}


if __name__ == '__main__':
    
    run_test=False
    plot_results=True
    load_res_path=None
 #   load_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ2\D1_AugAllfinal_resMinimal.csv"
   # load_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ2\D1a_AugStable_rolloff0.505_augment0.007_resMinimal.csv"
    load_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ2\D1a_AugStable_mergedTemp.csv"
    load_res_path=r"/home/michael/Downloads/D1a_AugStable_mergedTemp (1).csv"
    
    load_res_path=r"/home/michael/Downloads/D1b_RolloffStable_rolloff0.1_augment0.007_resMinimal.csv"
    
    load_res_path=r"/home/michael/Downloads/D1d_AugWarmstartfinal_resMinimal - Copy (1).csv"
    
    load_res_path=r"/home/michael/Downloads/D1_AugAllfinal_resMinimal - Copy (1).csv"
    
    load_res_warmstart=r"/home/michael/Downloads/D1d_AugWarmstartfinal_resMinimal - Copy (3).csv"
    
    load_res_path=r"/home/michael/Downloads/D1_AugAllfinal_resMinimal - Copy - Copy.csv"
    
    load_res_path=r"/home/michael/Downloads/D1_AugAllfinal_resMinimal - Copy - no SVM in ANY 0333NonSubj.csv"
    
    load_res_warmstart=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ2\D1d_AugWarmstartNewfinal_resMinimal.csv"
    
    load_res_warmstart=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ2\D1d_AugWarmstartfinal_resMinimal - Copy (3).csv"
    
    load_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ2\D1_AugAllfinal_resMinimal - Copy - Copy - Copy.csv"
    
    load_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ2\D1_AugAllfinal_resMinimal - Copy - no SVM in ANY 0333NonSubj.csv"
    
    load_res_warmstart=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ2\D1d_a_Warmstart_FixRFNewfinal_resMinimal - Copy.csv"

    systemUnderTest = 'D1a_AugStable'
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
        
    elif systemUnderTest == 'D1a_AugStable':
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
        augment_scales=[0,0.00666,0.02]#,0.1]#,0.67]
        '''try 0.1 maybe if time but 5mins x 10 trials x 20 subjects = 20h for one trainSize...'''
        # the scales above are 0, 1, 3, not 6, 7.89, not 12 (0.08), 25, 50, 100 per ppt per class
        # 0.05263 is 1/19, 7.89 per gest per ppt, i.e. result in aug_size = train_size
            #(actually ends up as 0.05333 = 8 per class per ppt = 152 in the aug)
        # 100 per class per ppt is the same amount as left over in the training set after 0.33 reserved for test
        # 50 and 100 removed for now for practicality as very big! dwarfs the subject
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
                    if rolloff==0.505 and np.isclose(augment_scale,0.006666666666):
                        skipRolloff=True
                        break
                    
                    if rolloff in [0.05,0.1,0.2575] and augment_scale < 0.1:
                        skipRolloff=True
                        break
                    
                    print('Rolloff: ',str(rolloff),' Augment: ',str(augment_scale))
                    
                    space=setup_search_space(architecture='decision',include_svm=True)
                    
                    space.update({'l1_sparsity':0.05})
                    space.update({'l1_maxfeats':40})
                    
                    space.update({'rolloff_factor':rolloff})
                    space.update({'augment_scale':augment_scale})
                    
                    
                    
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
                    
                    for repeat in range(10):
                        trials=Trials()
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
                        #winner_args['plot_confmats']=True
                        winner_args['subject id']=str(int(eeg_ppt['ID_pptID'][0]))
                        winner_args['repeat']=repeat
                        
                        subject_results=fusion_test(emg_joint,eeg_joint,emg_test,eeg_test,winner_args)
                        subject_results['best_loss']=best_loss
                        subject_results['repeat']=repeat
                        
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
                    
                results=results_final.join(winners_final.drop('repeat',axis=1))
                results['opt_acc']=1-results['best_loss']
                scores_minimal=results[['subject id','rolloff_factor','augment_scale','repeat','fusion_acc',
                               'emg_acc','eeg_acc','elapsed_time','fusion_alg_fusion_alg_type',
                               'eeg_eeg_model_type','emg_emg_model_type','featsel_method','train_method',
                               'best_loss','opt_acc']]
                
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
        per_ppt=0
        if per_ppt:
            for ppt in scores_minimal['subject id'].unique():
                subj= scores_minimal[scores_minimal['subject id']==ppt]
                subjScore=subj.groupby(['augment_scale','rolloff_factor'])['fusion_acc'].agg(['mean','std']).reset_index()
                if 0:
                    fig,ax=plt.subplots();
                    plt.rcParams['figure.dpi'] = 150 # DEFAULT IS 100
               #     for rolloff in np.sort(subj['rolloff_factor'].unique()):
               #             subj[subj['rolloff_factor']==rolloff].boxplot(column='fusion_acc',by='augment_scale',ax=ax)
               #     ax.set_ylim((0,1))
                    
              #      for key,group in subj.groupby('rolloff_factor'):
              #          grouped=group.groupby(['augment_scale'])['fusion_acc'].agg(['mean','std']).reset_index()
              #          plt.errorbar(x=grouped['augment_scale'],y=grouped['mean'],yerr=grouped['std'],
              #                       marker='.',capsize=5,label=key)
                    subjScore.pivot(index='augment_scale',columns='rolloff_factor',values='mean').plot(kind='bar',ax=ax,rot=0,capsize=5,
                                                                                                        yerr=subjScore.pivot(index='augment_scale',columns='rolloff_factor',values='std'))
                    ax.set_ylim(np.floor(subj['fusion_acc'].min()/0.05)*0.05,np.ceil(subj['fusion_acc'].max()/0.05)*0.05)
                    plt.title('Subject '+str(ppt))
                    ax.set_xlabel('Proportion of non-subject data augmenting')
                    ax.legend(title='Proportion of subject data')
                    plt.show()
                if 1:
                    fig,ax=plt.subplots();
                    plt.rcParams['figure.dpi'] = 150 # DEFAULT IS 100
                    subjScore.pivot(index='rolloff_factor',columns='augment_scale',values='mean').plot(kind='bar',ax=ax,rot=0,capsize=5,
                                                                                                        yerr=subjScore.pivot(index='rolloff_factor',columns='augment_scale',values='std'))
                    ax.set_ylim(np.floor(subj['fusion_acc'].min()/0.05)*0.05,np.ceil(subj['fusion_acc'].max()/0.05)*0.05)
                    plt.title('Subject '+str(ppt))
                    ax.set_xlabel('Proportion of subject data')
                    
                    plt.axhline(y=gen_dev_accs[int(ppt)],label='Generalist',linestyle='--',color='gray')
                    ax.legend(title='Proportion of non-subject data augmenting')
                    plt.show()
            
        
        
       
    
    '''
    fig,ax=plt.subplots();
    scores_agg=scores_minimal.groupby(['augment_scale','rolloff_factor'])['fusion_acc'].agg(['mean','std']).reset_index()
    scores_agg=scores_agg.round({'augment_scale':5})
    scores_agg.pivot(index='rolloff_factor',columns='augment_scale',values='mean').plot(kind='bar',ax=ax,rot=0,capsize=5,
                                                                                               yerr=scores_agg.pivot(index='rolloff_factor',columns='augment_scale',values='std'))
    ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
    plt.title('Means across subjects')
    ax.set_xlabel('Proportion of subject data')
    
    plt.axhline(y=0.723,label='Mean Generalist',linestyle='--',color='gray')
    ax.legend(title='Proportion non-subj augmenting')
    plt.show()
    '''
    plt.rcParams['figure.dpi']=150
    
    scores_aug=pd.read_csv(load_res_path,index_col=0)
    scores_xfer=pd.read_csv(load_res_warmstart,index_col=0)
    
    
    nSubj=19
    nGest=4
    nRepsPerGest=150
    nInstancePerGest=4
    trainsplitSize=2/3
    
    scores_aug['augscale_wholegests']=(np.around(scores_aug['augment_scale']*nSubj*nGest*nRepsPerGest)).astype(int)
    scores_xfer['augscale_wholegests']=(np.around(scores_xfer['augment_scale']*nSubj*nGest*nRepsPerGest)).astype(int)
    scores_aug['trainAmnt_wholegests']=(np.around(scores_aug['rolloff_factor']*trainsplitSize*nGest*nRepsPerGest)).astype(int)
    scores_xfer['trainAmnt_wholegests']=(np.around(scores_xfer['rolloff_factor']*trainsplitSize*nGest*nRepsPerGest)).astype(int)
    
    
    scores_aug_agg=scores_aug.groupby(['augscale_wholegests','trainAmnt_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
    scores_aug_agg=scores_aug_agg.round({'augscale_wholegests':5})
    scores_aug_agg=scores_aug_agg.add_prefix('aug_')
    scores_xfer_agg=scores_xfer.groupby(['augscale_wholegests','trainAmnt_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
    scores_xfer_agg=scores_xfer_agg.round({'augscale_wholegests':5})
    scores_xfer_agg=scores_xfer_agg.add_prefix('xfer_')
    
    figL,axL=plt.subplots()
    scores_aug_agg.pivot(columns='aug_trainAmnt_wholegests',
                     index='aug_augscale_wholegests',
                     values='aug_mean').plot(kind='line',marker='.',ax=axL,rot=0)
    plt.gca().set_prop_cycle(None)
    scores_xfer_agg.pivot(columns='xfer_trainAmnt_wholegests',
                     index='xfer_augscale_wholegests',
                     values='xfer_mean').plot(kind='line',marker='x',linestyle='-.',ax=axL,rot=0)
    plt.title('Means across ppts on reserved 33% (200 gestures)')
    axL.set_xlabel('# Non-subject gestures (of 11400)')
    axL.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
    
    plt.axhline(y=0.723,label='Mean\nGeneralist',linestyle='--',color='gray')
    plt.axhline(y=0.866105,label='Fully\nBespoke',linestyle='--',color='pink')
    #handles,labels=axL.get_legend_handles_labels()
    #plt.tight_layout()
    axL.legend(title='# Subject gestures\n (of 400)',loc='center left',bbox_to_anchor=(1,0.5))
    #axL.set_ylim(np.floor(scores_agg[scores_agg['mean']>0]['mean'].min()/0.1)*0.1,np.ceil(scores_agg['mean'].max()/0.05)*0.05)
    #axL.set_ylim(0.6,0.9)
    plt.show()
    
    
    figL,axL=plt.subplots()
    scores_aug_agg.pivot(index='aug_trainAmnt_wholegests',
                     columns='aug_augscale_wholegests',
                     values='aug_mean').plot(kind='line',marker='.',ax=axL,rot=0)
    plt.gca().set_prop_cycle(None)
    next(axL._get_lines.prop_cycler)

    
    scores_xfer_agg.pivot(index='xfer_trainAmnt_wholegests',
                     columns='xfer_augscale_wholegests',
                     values='xfer_mean').plot(kind='line',marker='x',linestyle='-.',ax=axL,rot=0)
    plt.title('Means across ppts on reserved 33% (200 gestures)')
    axL.set_xlabel('# Subject gestures (of 400)')
    axL.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
    
    plt.axhline(y=0.723,label='Mean\nGeneralist',linestyle='--',color='gray')
    plt.axhline(y=0.866105,label='Fully\nBespoke',linestyle='--',color='pink')
    #handles,labels=axL.get_legend_handles_labels()
    #plt.tight_layout()
    axL.legend(title='# Non-subject gestures (of 11400)',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
    #axL.set_ylim(np.floor(scores_agg[scores_agg['mean']>0]['mean'].min()/0.1)*0.1,np.ceil(scores_agg['mean'].max()/0.05)*0.05)
    axL.set_ylim(0.7,0.9)
    plt.show()
    
    
    
    
    figL,axL=plt.subplots()
    scores_aug_agg.pivot(columns='aug_trainAmnt_wholegests',
                     index='aug_augscale_wholegests',
                     values='aug_mean').plot(kind='line',marker='.',ax=axL,rot=0)
    plt.gca().set_prop_cycle(None)
    scores_xfer_agg.pivot(columns='xfer_trainAmnt_wholegests',
                     index='xfer_augscale_wholegests',
                     values='xfer_mean').plot(kind='line',marker='x',linestyle='-.',ax=axL,rot=0)
    plt.title('Means across ppts on reserved 33% (200 gestures)')
    axL.set_xlabel('# Non-subject gestures (of 11400)')
    axL.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
    
    plt.axhline(y=0.723,label='Mean\nGeneralist',linestyle='--',color='gray')
    plt.axhline(y=0.866105,label='Fully\nBespoke',linestyle='--',color='pink')
    #handles,labels=axL.get_legend_handles_labels()
    #plt.tight_layout()
    axL.legend(title='# Subject gestures\n (of 400)',loc='center left',bbox_to_anchor=(1,0.5))
    #axL.set_ylim(np.floor(scores_agg[scores_agg['mean']>0]['mean'].min()/0.1)*0.1,np.ceil(scores_agg['mean'].max()/0.05)*0.05)
    axL.set_ylim(0.6,0.9)
    axL.set_xlim(-100,2000)
    plt.show()
    
    
    figL,axL=plt.subplots()
    scores_aug_agg=scores_aug_agg.pivot(columns='aug_trainAmnt_wholegests',
                     index='aug_augscale_wholegests',
                     values='aug_mean')
   # scores_aug_agg=np.log(scores_aug_agg)
    scores_aug_agg.plot(kind='line',marker='.',ax=axL,rot=0)
    #scores_aug_agg=scores_aug_agg.add_suffix(' ')
    plt.gca().set_prop_cycle(None)
    scores_xfer_agg=scores_xfer_agg.pivot(columns='xfer_trainAmnt_wholegests',
                     index='xfer_augscale_wholegests',
                     values='xfer_mean')
   # scores_xfer_agg=np.log(scores_xfer_agg)
    scores_xfer_agg.plot(kind='line',marker='x',linestyle='-.',ax=axL,rot=0)
    plt.title('Means across Development subjects on reserved 33% (200 gestures)',loc='left')
    axL.set_xlabel('# Other-Subject gestures (of 11400)')
    axL.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
    
    plt.axhline(y=0.723,label='Generalist*',linestyle='--',color='gray')
  #  plt.axhline(y=0.866105,label='Fully Bespoke',linestyle='--',color='pink')
    #handles,labels=axL.get_legend_handles_labels()
    #plt.tight_layout()
    lines=axL.get_lines()
    h,l=axL.get_legend_handles_labels()
    legendColours=plt.legend(h[0:6],l[0:6],title='# Same-Subject\ngestures (of 400)',loc='upper left',bbox_to_anchor=(1,0.5))
  #  legendSystem=plt.legend([h[i] for i in [0,6,12,13]],['Augmentation','Model Transfer']+l[12:],title='System type',loc='lower left',bbox_to_anchor=(1,0.5))
    legendSystem=plt.legend([h[i] for i in [0,6,12]],['Augmentation','Model Transfer']+l[12:],title='System type',loc='lower left',bbox_to_anchor=(1,0.5))
    #axL.legend(title='# Subject gestures\n (of 400)',loc='center left',bbox_to_anchor=(1,0.5))
    axL.add_artist(legendColours)
    axL.add_artist(legendSystem)
    #axL.set_ylim(np.floor(scores_agg[scores_agg['mean']>0]['mean'].min()/0.1)*0.1,np.ceil(scores_agg['mean'].max()/0.05)*0.05)
    axL.set_ylim(0.43,0.93)
  #  axL.set_yscale('function', functions=(partial(np.power, 10.0), np.log10))
  #  axL.set_ylim(0.43,0.9)
    #axL.set_xlim(-100,2000)
    plt.show()
    
    
    
    fig,ax=plt.subplots()
    aug_301_dev=scores_aug_agg[scores_aug_agg['aug_trainAmnt_wholegests']==301]
    xfer_301_dev=scores_xfer_agg[scores_xfer_agg['xfer_trainAmnt_wholegests']==301]
    
    aug_301_dev.pivot(index='aug_augscale_wholegests',
                     columns='aug_trainAmnt_wholegests',
                     values='aug_mean').plot(kind='line',ax=ax,rot=0,marker='.',linestyle='-',color='tab:purple'
                                         #yerr=scores_aug_agg.pivot(index='augscale_wholegests',columns='trainAmnt_wholegests',values='std'),
                                         )
    xfer_301_dev.pivot(index='xfer_augscale_wholegests',
                     columns='xfer_trainAmnt_wholegests',
                     values='xfer_mean').plot(kind='line',ax=ax,rot=0,marker='x',linestyle='-.',color='tab:purple'
                                         #yerr=scores_agg.pivot(index='augscale_wholegests',columns='trainAmnt_wholegests',values='std'),
                                         )
    plt.title('Means across Development subjects on reserved 33% (200 gestures)\n    where system has access to 301 same-subject gestures',loc='left')
    ax.set_xlabel('# Other-Subject gestures (max 12000)')
    ax.set_ylabel('Classification Accuracy')
    
    ax.set_ylim(0.455,0.88)
    ax.set_ylim(0.43,0.93)
    
    plt.axhline(y=0.723,label='RQ1 Generalist',linestyle='--',color='gray')
    h,l=ax.get_legend_handles_labels()
    
    ax.legend(h,['Augmentation','Transfer Learning','RQ1 Generalist'],title='System',loc='center left',bbox_to_anchor=(1,0.5))
    plt.show()
    
    
    raise
    
    
    scores_minimal['augscale_instances']=scores_minimal['augment_scale']*nSubj*nGest*nRepsPerGest*nInstancePerGest
    scores_minimal['augscale_wholegests']=(np.around(scores_minimal['augment_scale']*nSubj*nGest*nRepsPerGest)).astype(int)
    scores_minimal['augscale_pergest']=scores_minimal['augment_scale']*nSubj*nRepsPerGest
    scores_minimal['augscale_pergestpersubj']=scores_minimal['augment_scale']*nRepsPerGest
    
    scores_minimal['trainAmnt_instances']=scores_minimal['rolloff_factor']*(1-testset_size)*nGest*nRepsPerGest*nInstancePerGest
    #scores_minimal['trainAmnt_wholegests']=scores_minimal['rolloff_factor']*(1-testset_size)*nGest*nRepsPerGest
    scores_minimal['trainAmnt_wholegests']=(np.around(scores_minimal['rolloff_factor']*trainsplitSize*nGest*nRepsPerGest)).astype(int)
    scores_minimal['trainAmnt_pergest']=scores_minimal['rolloff_factor']*(1-testset_size)*nRepsPerGest
    
    
    fig,ax=plt.subplots();
    scores_agg=scores_minimal.groupby(['augscale_wholegests','trainAmnt_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
    scores_agg=scores_agg.round({'augscale_wholegests':5})
    scores_agg.pivot(index='trainAmnt_wholegests',
                     columns='augscale_wholegests',
                     values='mean').plot(kind='bar',ax=ax,rot=0,capsize=2,width=0.8,
                                         yerr=scores_agg.pivot(index='trainAmnt_wholegests',
                                                               columns='augscale_wholegests',values='std'))
    ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
    plt.title('Means across ppts on reserved 33% (200 gestures), Aug')
    ax.set_xlabel('# Subject gestures present (max 400)')
    ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
    
    plt.axhline(y=0.723,label='Mean\nGeneralist',linestyle='--',color='gray')
    plt.axhline(y=0.866105,label='Fully\nBespoke',linestyle='--',color='pink')
    plt.axhline(y=0.866105-0.047168,linestyle=':',color='pink')
    plt.axhline(y=0.866105+0.047168,linestyle=':',color='pink')
    plt.axhline(y=0.723-0.073289,linestyle=':',color='gray')
    plt.axhline(y=0.723+0.073289,linestyle=':',color='gray')
    ax.legend(title='# Non-subject gestures\n (max 11400)',loc='center left',bbox_to_anchor=(1,0.5))
    plt.show()
    
    figL,axL=plt.subplots()
    scores_agg.pivot(index='trainAmnt_wholegests',
                     columns='augscale_wholegests',
                     values='mean').plot(kind='line',marker='.',ax=axL,rot=0)
    plt.title('Means across ppts on reserved 33% (200 gestures), Aug')
    axL.set_xlabel('# Subject gestures present (of 400)')
    axL.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
    
    plt.axhline(y=0.723,label='Mean\nGeneralist',linestyle='--',color='gray')
    plt.axhline(y=0.866105,label='Fully\nBespoke',linestyle='--',color='pink')
    #plt.tight_layout()
    axL.legend(title='# Non-subject gestures (of 11400)',loc='center left',bbox_to_anchor=(1,0.5))
    axL.set_ylim(np.floor(scores_agg[scores_agg['mean']>0]['mean'].min()/0.1)*0.1,np.ceil(scores_agg['mean'].max()/0.05)*0.05)
    plt.show()
    
   
    fig,ax=plt.subplots();
    scores_agg=scores_minimal.groupby(['augscale_wholegests','trainAmnt_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
    scores_agg=scores_agg.round({'trainAmnt_wholegests':5})
    scores_agg.pivot(index='augscale_wholegests',
                     columns='trainAmnt_wholegests',
                     values='mean').plot(kind='bar',ax=ax,rot=0,capsize=2,width=0.8,
                                         yerr=scores_agg.pivot(index='augscale_wholegests',
                                                               columns='trainAmnt_wholegests',values='std'))
    ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
    plt.title('Means across ppts on reserved 33% (200 gestures), Aug')
    ax.set_xlabel('# Non-subject gestures (max 11400)')
    ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
    
    plt.axhline(y=0.723,label='Mean\nGeneralist',linestyle='--',color='gray')
    plt.axhline(y=0.866105,label='Fully\nBespoke',linestyle='--',color='pink')
    plt.axhline(y=0.866105-0.047168,linestyle=':',color='pink')
    plt.axhline(y=0.866105+0.047168,linestyle=':',color='pink')
    plt.axhline(y=0.723-0.073289,linestyle=':',color='gray')
    plt.axhline(y=0.723+0.073289,linestyle=':',color='gray')
    ax.legend(title='# Subject gestures\n present (max 400)',loc='center left',bbox_to_anchor=(1,0.5))
    plt.show()
    
    
    figL,axL=plt.subplots()
    scores_agg.pivot(columns='trainAmnt_wholegests',
                     index='augscale_wholegests',
                     values='mean').plot(kind='line',marker='.',ax=axL,rot=0)
    plt.title('Means across ppts on reserved 33% (200 gestures), Aug')
    axL.set_xlabel('# Non-subject gestures (of 11400)')
    axL.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
    
    plt.axhline(y=0.723,label='Mean\nGeneralist',linestyle='--',color='gray')
    plt.axhline(y=0.866105,label='Fully\nBespoke',linestyle='--',color='pink')
    #handles,labels=axL.get_legend_handles_labels()
    #plt.tight_layout()
    axL.legend(title='# Subject gestures\n present (of 400)',loc='center left',bbox_to_anchor=(1,0.5))
    axL.set_ylim(np.floor(scores_agg[scores_agg['mean']>0]['mean'].min()/0.1)*0.1,np.ceil(scores_agg['mean'].max()/0.05)*0.05)
    plt.show()
    
    scores_aug_agg=scores_agg.copy()
    if 1:   
        scores_minimal=pd.read_csv(load_res_warmstart,index_col=0)
        
        nSubj=19
        nGest=4
        nRepsPerGest=150
        nInstancePerGest=4
        trainsplitSize=2/3
        scores_minimal['augscale_instances']=scores_minimal['augment_scale']*nSubj*nGest*nRepsPerGest*nInstancePerGest
        scores_minimal['augscale_wholegests']=(np.around(scores_minimal['augment_scale']*nSubj*nGest*nRepsPerGest)).astype(int)
        scores_minimal['augscale_pergest']=scores_minimal['augment_scale']*nSubj*nRepsPerGest
        scores_minimal['augscale_pergestpersubj']=scores_minimal['augment_scale']*nRepsPerGest
        
        scores_minimal['trainAmnt_instances']=scores_minimal['rolloff_factor']*(1-testset_size)*nGest*nRepsPerGest*nInstancePerGest
        #scores_minimal['trainAmnt_wholegests']=scores_minimal['rolloff_factor']*(1-testset_size)*nGest*nRepsPerGest
        scores_minimal['trainAmnt_wholegests']=(np.around(scores_minimal['rolloff_factor']*trainsplitSize*nGest*nRepsPerGest)).astype(int)
        scores_minimal['trainAmnt_pergest']=scores_minimal['rolloff_factor']*(1-testset_size)*nRepsPerGest
        
        '''add dummy data of nonsubj=0 to line up the colour cycle'''
        dummy_data=np.zeros((1,len(scores_minimal.columns.values)),dtype=int)
        scores_minimal=scores_minimal.append(pd.DataFrame(data=dummy_data,columns=scores_minimal.columns))
        
        fig,ax=plt.subplots();
        scores_agg=scores_minimal.groupby(['augscale_wholegests','trainAmnt_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_agg=scores_agg.round({'augscale_wholegests':5})
        scores_agg.pivot(index='trainAmnt_wholegests',
                         columns='augscale_wholegests',
                         values='mean').plot(kind='bar',ax=ax,rot=0,capsize=2,width=0.8,
                                             yerr=scores_agg.pivot(index='trainAmnt_wholegests',
                                                                   columns='augscale_wholegests',values='std'))
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Means across ppts on reserved 33% (200 gestures), Model Xfer')
        ax.set_xlabel('# Subject gestures calibrating (of 400)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        plt.axhline(y=0.723,label='Mean\nGeneralist',linestyle='--',color='gray')
        plt.axhline(y=0.866105,label='Fully\nBespoke',linestyle='--',color='pink')
        plt.axhline(y=0.866105-0.047168,linestyle=':',color='pink')
        plt.axhline(y=0.866105+0.047168,linestyle=':',color='pink')
        plt.axhline(y=0.723-0.073289,linestyle=':',color='gray')
        plt.axhline(y=0.723+0.073289,linestyle=':',color='gray')
        '''mean and std dev here are calculated from D1a_AugStable_rolloff1.0_resMinimal
        i.e. from 10 repeats per subject of no rolloff, no aug, fully bespoke'''
        '''but note this is from across subjects. for each subject, stddev of no rolloff
        no aug is typically 0.014, ranges from 0.004 to 0.033'''
        ax.legend(title='# Non-subject gestures\n training (of 11400)',loc='center left',bbox_to_anchor=(1,0.5))
        left,right=ax.get_xlim()
        ax.set_xlim(left-1.5*left,right)
        handles,labels=ax.get_legend_handles_labels()
        #plt.tight_layout()
        ax.legend(handles[0:2]+handles[3:],labels[0:2]+labels[3:],title='# Non-subject gestures\n training (of 11400)',loc='center left',bbox_to_anchor=(1,0.5))
        plt.show()
        
        
        figL,axL=plt.subplots()
        scores_agg.pivot(index='trainAmnt_wholegests',
                         columns='augscale_wholegests',
                         values='mean').plot(kind='line',marker='.',ax=axL,rot=0)
        plt.title('Means across ppts on reserved 33% (200 gestures), Model Xfer')
        axL.set_xlabel('# Subject gestures calibrating (of 400)')
        axL.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        plt.axhline(y=0.723,label='Mean\nGeneralist',linestyle='--',color='gray')
        plt.axhline(y=0.866105,label='Fully\nBespoke',linestyle='--',color='pink')
        handles,labels=axL.get_legend_handles_labels()
        #plt.tight_layout()
        axL.legend(handles[1:],labels[1:],title='# Non-subject gestures\n training (of 11400)',loc='center left',bbox_to_anchor=(1,0.5))
        axL.set_ylim(np.floor(scores_agg[scores_agg['mean']>0]['mean'].min()/0.1)*0.1,np.ceil(scores_agg['mean'].max()/0.05)*0.05)
        plt.show()
        
        
        
        
        '''remove the dummy data again'''
        scores_minimal=scores_minimal[~scores_minimal.eq(0).all(axis=1)]
        
        fig,ax=plt.subplots();
        scores_agg=scores_minimal.groupby(['augscale_wholegests','trainAmnt_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_agg=scores_agg.round({'trainAmnt_wholegests':5})
        scores_agg.pivot(index='augscale_wholegests',
                         columns='trainAmnt_wholegests',
                         values='mean').plot(kind='bar',ax=ax,rot=0,capsize=2,width=0.8,
                                             yerr=scores_agg.pivot(index='augscale_wholegests',
                                                                   columns='trainAmnt_wholegests',values='std'))
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Means across ppts on reserved 33% (200 gestures), Model Xfer')
        ax.set_xlabel('# Non-subject gestures initial training (of 11400)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        plt.axhline(y=0.723,label='Mean\nGeneralist',linestyle='--',color='gray')
        plt.axhline(y=0.866105,label='Fully\nBespoke',linestyle='--',color='pink')
        plt.axhline(y=0.866105-0.047168,linestyle=':',color='pink')
        plt.axhline(y=0.866105+0.047168,linestyle=':',color='pink')
        plt.axhline(y=0.723-0.073289,linestyle=':',color='gray')
        plt.axhline(y=0.723+0.073289,linestyle=':',color='gray')
        ax.legend(title='# Subject gestures\n calibrating (of 400)',loc='center left',bbox_to_anchor=(1,0.5))
        #left,right=ax.get_xlim()
        #ax.set_xlim(left-1.5*left,right)
        #plt.tight_layout()
        plt.show()
        
        
        figL,axL=plt.subplots()
        scores_agg.pivot(columns='trainAmnt_wholegests',
                         index='augscale_wholegests',
                         values='mean').plot(kind='line',marker='.',ax=axL,rot=0)
        plt.title('Means across ppts on reserved 33% (200 gestures), Model Xfer')
        axL.set_xlabel('# Non-subject gestures training (of 11400)')
        axL.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        plt.axhline(y=0.723,label='Mean\nGeneralist',linestyle='--',color='gray')
        plt.axhline(y=0.866105,label='Fully\nBespoke',linestyle='--',color='pink')
        #handles,labels=axL.get_legend_handles_labels()
        #plt.tight_layout()
        axL.legend(title='# Subject gestures\ncalibrating (of 400)',loc='center left',bbox_to_anchor=(1,0.5))
        axL.set_ylim(np.floor(scores_agg[scores_agg['mean']>0]['mean'].min()/0.1)*0.1,np.ceil(scores_agg['mean'].max()/0.05)*0.05)
        plt.show()
        
        
        
        
    
if 0:
    def load_results_obj(path):
        load_trials=pickle.load(open(path,'rb'))
        load_table=pd.DataFrame(load_trials.trials)
        load_table_readable=pd.concat(
            [pd.DataFrame(load_table['result'].tolist()),
             pd.DataFrame(pd.DataFrame(load_table['misc'].tolist())['vals'].values.tolist())],
            axis=1,join='outer')
        return load_trials,load_table,load_table_readable
    
    _,_,gen_results=load_results_obj(r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Generalist_20DevSet\Gen_feat_joint\trials_obj.p")
    gen_best=gen_results.iloc[76]
    gen_best_accs=gen_best['fusion_accs']
    gen_dev_accs=dict(zip(scores_minimal['subject id'].unique(),gen_best_accs))
    mean_gen_acc=np.mean(np.array([*gen_dev_accs.values()]))
    std_gen_acc=np.std(np.array([*gen_dev_accs.values()]))
    

    