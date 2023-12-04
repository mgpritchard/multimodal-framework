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
    
    if args['calib_level'] < 8/134:
        raise ValueError('Insufficient data top optimise')
    elif args['calib_level']==8/134:
        testset_size=0.5
    else:
        testset_size=args['testset_size']
    
    train_split,test_split=train_test_split(gest_strat,test_size=testset_size,
                                            random_state=random_split,stratify=gest_strat[1])
    
    
    eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split[0])]
    emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]       
            
    eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(train_split[0])]
    emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(train_split[0])]
    


    if len(emg_test['Label'].value_counts())<4:
        raise ValueError('Not all gestures in optimisation-test split')
    
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



gen_dev_accs={2: 0.76625, 3: 0.68875, 4: 0.7879166666666667, 5: 0.77875, 7: 0.81, 8: 0.745, 9: 0.6504166666666666,
              10: 0.6991666666666667, 12: 0.70375, 13: 0.5275, 14: 0.6008333333333333, 15: 0.65875,
              17: 0.8183333333333334, 18: 0.74875, 19: 0.7379166666666667, 20: 0.7408333333333333,
              22: 0.7375, 23: 0.6825, 24: 0.8375, 25: 0.7395833333333334}

def emgsearchspace_lowdata(max_knn_samples,include_svm=True):
    emgoptions=[
                {'emg_model_type':'RF',
                 'n_trees':scope.int(hp.quniform('emg.RF.ntrees',10,100,q=5)),
                 'max_depth':5,#scope.int(hp.quniform('emg.RF.maxdepth',2,5,q=1)),
                 #integerising search space https://github.com/hyperopt/hyperopt/issues/566#issuecomment-549510376
                 },
                {'emg_model_type':'kNN',
                 'knn_k':scope.int(hp.quniform('emg.knn.k',1,max_knn_samples,q=1)),
                 },
                {'emg_model_type':'LDA',
                 'LDA_solver':hp.choice('emg.LDA_solver',['svd','lsqr','eigen']), #removed lsqr due to LinAlgError: SVD did not converge in linear least squares but readding as this has not repeated
                 'shrinkage':hp.uniform('emg.lda.shrinkage',0.0,0.5),
                 },
                {'emg_model_type':'QDA', #emg qda reg 0.3979267 actually worked well!! for withinppt
                 'regularisation':hp.uniform('emg.qda.regularisation',0.0,1.0), #https://www.kaggle.com/code/code1110/best-parameter-s-for-qda/notebook
                 },
                ]
    if include_svm:
        emgoptions.append({'emg_model_type':'SVM_PlattScale',
                 'kernel':hp.choice('emg.svm.kernel',['rbf']),#'poly','linear']),
                 'svm_C':hp.loguniform('emg.svm.c',np.log(0.1),np.log(100)), #use loguniform? #https://queirozf.com/entries/choosing-c-hyperparameter-for-svm-classifiers-examples-with-scikit-learn
                 'gamma':hp.loguniform('emg.svm.gamma',np.log(0.01),np.log(0.2)), #maybe log, from lower? #https://vitalflux.com/svm-rbf-kernel-parameters-code-sample/
                 #eg sklearns gridsearch doc uses SVC as an example with C log(1e0,1e3) & gamma log(1e-4,1e-3)
                 })
    newemgspace = (hp.choice('emg model',emgoptions))
    return newemgspace

def eegsearchspace_lowdata(include_svm=True):
    eegoptions=[
                {'eeg_model_type':'RF',
                 'n_trees':scope.int(hp.quniform('eeg_ntrees',10,100,q=5)),
                 'max_depth':5,#scope.int(hp.quniform('eeg.RF.maxdepth',2,5,q=1)),
                 },
                {'eeg_model_type':'LDA',
                 'LDA_solver':hp.choice('eeg.LDA_solver',['svd','lsqr','eigen']),
                 'shrinkage':hp.uniform('eeg.lda.shrinkage',0.0,0.5),
                 },
                {'eeg_model_type':'QDA',
                 'regularisation':hp.uniform('eeg.qda.regularisation',0.0,1.0),
                 },
                ]
    if include_svm:
        eegoptions.append({'eeg_model_type':'SVM_PlattScale',
                 'kernel':hp.choice('eeg.svm.kernel',['rbf']),#'poly','linear']),
                 'svm_C':hp.loguniform('eeg.svm.c',np.log(0.01),np.log(100)),
                 'gamma':hp.loguniform('eeg.svm.gamma',np.log(0.01),np.log(0.2)), 
                 #https://www.kaggle.com/code/donkeys/exploring-hyperopt-parameter-tuning?scriptVersionId=12655875&cellId=64
                 # naming convention https://github.com/hyperopt/hyperopt/issues/380#issuecomment-685173200
                 })
    neweegspace = (hp.choice('eeg model',eegoptions))
    return neweegspace

if __name__ == '__main__':
    
    run_test=False
    plot_results=True
    load_res_path=None
    
    load_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\C1_Session3onlyfinal_resMinimal - Copy.csv"
    #load_res_path=r"/home/michael/Downloads/C1_Session3onlyfinal_resMinimal - Copy.csv"
    load_calib_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\B1_AugPipelinefinal_resMinimal - Copy.csv"
    #load_calib_res_path=r"/home/michael/Downloads/B1_AugPipelinefinal_resMinimal - Copy.csv"
    
    systemUnderTest = 'C1_Session3only'
    
    if systemUnderTest=='A1a_session1':
        train_session='first'
    elif systemUnderTest=='A1b_session2':
        train_session='second'
    elif systemUnderTest=='A2_both1and2':
        train_session='both'
    
    feats_method='no calib'
    opt_method='no calib'
    train_method='no calib'
    
    calib_levels = [0]
    
    if systemUnderTest=='B1_AugPipeline':
        feats_method='calib'
        opt_method='calib'
        train_method='calib'
        
        train_session='both'
        
        calib_levels = [4/134,8/134,20/134,40/134,100/134,132/134]
        # above is 1 2 5 10 25 33 performances of each gesture from session3
        
        # 1 is maybe not doable as cant optimise for it
        calib_levels = [20/134,40/134,100/134,132/134]
        
        calib_levels=[60/134,72/134,120/134]
        
        calib_levels = np.array([round(scale/(4/134))*(4/134) for scale in calib_levels])
        
    if systemUnderTest=='C1_Session3only':
        feats_method='no calib'
        opt_method='no calib'
        train_method='no calib'
        
        train_session='third'
        
        calib_levels = [8/134,20/134,40/134,60/134,72/134,80/134,100/134,120/134,132/134]
        
        calib_levels = np.array([round(scale/(4/134))*(4/134) for scale in calib_levels])
    
    testset_size = 0.33
    
    
    
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
        
        for calib_level in calib_levels:
                for idx,emg_mask in enumerate(emg_masks):
                    print('Session3only level: ',str(calib_level),' (subject ',str(idx),' of 20)')
                    
                    space=setup_search_space(architecture='decision',include_svm=True)
                    
                    if calib_level==8/134:
                        '''handling rare case where session 3 traindata is so low that when
                        trying to train a knn *within a split for metamodel-fusion training
                        there are insufficient samples for the value of k chosen'''
                        
                        '''ie because 200 session3, 67% of that = 134 training, reduced to 8/134 ie 8
                        those 8 are split 4--4 within optimisation, so the opt can train on only 4 (16 instances).
                        a stacked fusion alg splits that into 3 folds, uses ~2 for training (so 10) and 6 to learn
                        predictions to train the meta-fuser with.
                        Therefore only 10 datapoints has to be max kkn K as cant have more neighbours than data'''
                        #max_knn_samples=calib_level*134*0.67*4*0.67 ?
                        
                        '''but ALSO have to discount SVMs as they rely on 5-fold CV, insufficient data to 5fold'''
                        space.update({'emg':emgsearchspace_lowdata(max_knn_samples=10,include_svm=False),
                                      'eeg':eegsearchspace_lowdata(include_svm=False)})
                    
                    # approx 134* calib_level (x/134) * 0.67 (split within opt, for 8/134 this was 0.5) * 4 gestures *0.67 for two metamodel-training folds
                    # therefore if 12/134 tried, would be max of 21
                    # with 16/134 would be all good at 28 (knn is already max of 25)
                        
                    
                    
                    space.update({'l1_sparsity':0.05})
                    space.update({'l1_maxfeats':40})
                    
                    space.update({'calib_level':calib_level})
                    
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
                    emg_ppt['ID_stratID']=emg_ppt['ID_run'].astype(str)+emg_ppt['Label'].astype(str)+emg_ppt['ID_gestrep'].astype(str)
                    random_split=random.randint(0,100)
                    
                    if not emg_ppt['ID_stratID'].equals(eeg_ppt['ID_stratID']):
                        raise ValueError('EMG & EEG performances misaligned')
                        
                        
                    #emg_session1=emg_ppt[emg_ppt['ID_run']==1.0]
                    #eeg_session1=eeg_ppt[eeg_ppt['ID_run']==1.0]
                    #emg_session2=emg_ppt[emg_ppt['ID_run']==2.0]
                    #eeg_session2=eeg_ppt[eeg_ppt['ID_run']==2.0]
                    emg_session3=emg_ppt[emg_ppt['ID_run']==3.0]
                    eeg_session3=eeg_ppt[eeg_ppt['ID_run']==3.0]
                         
                
                    gest_perfs=emg_session3['ID_stratID'].unique()
                    gest_strat=pd.DataFrame([gest_perfs,[perf.split('.')[1][-1] for perf in gest_perfs]]).transpose()
                    
                    train_split,test_split=train_test_split(gest_strat,test_size=testset_size,
                                                          random_state=random_split,stratify=gest_strat[1])
                                   
                    
                    if calib_level == 1:
                        train_split=train_split
                    else:
                        if space['calib_level'] > 130/134:
                            '''case for when we cant train_test_split as the (unused) "test" would be < n_classes'''
                            stratsize=np.min(train_split[1].value_counts())
                            train_split = train_split.groupby(1,axis=0)
                            train_split=train_split.apply(lambda x: x.sample(stratsize))
                        elif 0 < space['calib_level'] < 1:
                            train_split,_=train_test_split(train_split,train_size=space['calib_level'],random_state=random_split,stratify=train_split[1])
                            if min(train_split[1].value_counts()) < 2:
                                print('session3 level of ' +str(space['calib_level'])+' results in < 2 performances per class')
                                skipRolloff=True
                                break
                    
                    emg_train=emg_session3[emg_session3['ID_stratID'].isin(train_split[0])]
                    eeg_train=eeg_session3[eeg_session3['ID_stratID'].isin(train_split[0])]
                    
                    emg_train,emgscaler=feats.scale_feats_train(emg_train,space['scalingtype'])
                    eeg_train,eegscaler=feats.scale_feats_train(eeg_train,space['scalingtype'])
                    
                    emg_test=emg_session3[emg_session3['ID_stratID'].isin(test_split[0])]
                    eeg_test=eeg_session3[eeg_session3['ID_stratID'].isin(test_split[0])]
                    
                    emg_test=feats.scale_feats_test(emg_test,emgscaler)
                    eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
                    

                    sel_cols_emg=feats.sel_percent_feats_df(ml.drop_ID_cols(emg_train),percent=15)
                    sel_cols_emg=np.append(sel_cols_emg,ml.drop_ID_cols(emg_train).columns.get_loc('Label'))
                    sel_cols_eeg=feats.sel_feats_l1_df(ml.drop_ID_cols(eeg_train),sparsityC=space['l1_sparsity'],maxfeats=space['l1_maxfeats'])
                    sel_cols_eeg=np.append(sel_cols_eeg,ml.drop_ID_cols(eeg_train).columns.get_loc('Label')) 

                
 
                    space['sel_cols_emg']=sel_cols_emg
                    space['sel_cols_eeg']=sel_cols_eeg
                    space['subject-id']=eeg_ppt['ID_pptID'][0]  
                    
                    space.update({'emg_set':emg_train,'eeg_set':eeg_train,'data_in_memory':True,'prebalanced':True})

                    space.update({'featsel_method':feats_method})
                    space.update({'train_method':train_method})
                    space.update({'opt_method':opt_method})
                    space.update({'train_session':train_session})
                    
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
                    
                    subject_results=fusion_test(emg_train,eeg_train,emg_test,eeg_test,winner_args)
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
                                                  'scalingtype','plot_confmats','l1_maxfeats','l1_sparsity','get_train_acc',],axis=1)    
                #winners_final=pd.concat([winners_final.eeg.apply(pd.Series), winners_final.drop('eeg', axis=1)], axis=1)
                #winners_final=pd.concat([winners_final.emg.apply(pd.Series), winners_final.drop('emg', axis=1)], axis=1)
                #winners_final=pd.concat([winners_final.fusion_alg.apply(pd.Series), winners_final.drop('fusion_alg', axis=1)], axis=1)
                    
                for col in ['emg','eeg','fusion_alg']:
                    winners_final=pd.concat([winners_final.drop(col,axis=1),
                                             winners_final[col].apply(lambda x: pd.Series(x)).rename(columns=lambda x: f'{col}_{x}')],axis=1)    
                    
                results=results_final.join(winners_final)
                results['opt_acc']=1-results['best_loss']
                scores_minimal=results[['subject id','fusion_acc','emg_acc','eeg_acc','elapsed_time','train_session',
                               'fusion_alg_fusion_alg_type','eeg_eeg_model_type','emg_emg_model_type',
                               'featsel_method','opt_method','train_method','calib_level','best_loss','opt_acc']]
                
                currentpath=os.path.dirname(__file__)
                result_dir=params.jeong_results_dir
                resultpath=os.path.join(currentpath,result_dir)    
                resultpath=os.path.join(resultpath,'RQ3')
               
                
                
                if calib_level==0:
                    calib_label=''
                else:
                    calib_label='_calib'+str(round(calib_level,3))
                
                picklepath=os.path.join(resultpath,(systemUnderTest+calib_label+'_resDF.pkl'))
                csvpath=os.path.join(resultpath,(systemUnderTest+calib_label+'_resMinimal.csv'))
        
                pickle.dump(results,open(picklepath,'wb'))
                scores_minimal.to_csv(csvpath)
        
        picklefullpath=os.path.join(resultpath,(systemUnderTest+'final_resDF.pkl'))
        csvfullpath=os.path.join(resultpath,(systemUnderTest+'final_resMinimal.csv'))

        pickle.dump(results,open(picklefullpath,'wb'))
        scores_minimal.to_csv(csvfullpath)
    else:
        scores_minimal=pd.read_csv(load_res_path,index_col=0)
        
    if plot_results:
        scores_calib_minimal=pd.read_csv(load_calib_res_path,index_col=0)
        '''

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
        
        '''removing ppt 14 who may be outlier'''
        #scores_minimal=scores_minimal[scores_minimal['subject id']!=14]
        
        nGest=4
        nRepsPerGest=50
        nInstancePerGest=4
        trainsplitSize=2/3
        scores_minimal['calib_level_instances']=scores_minimal['calib_level']*(1-testset_size)*nGest*nRepsPerGest*nInstancePerGest
        scores_minimal['calib_level_wholegests']=scores_minimal['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        scores_minimal['calib_level_pergest']=scores_minimal['calib_level']*(1-testset_size)*nRepsPerGest
        
        scores_calib_minimal['calib_level_wholegests']=scores_calib_minimal['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        
        
        
        fig,ax=plt.subplots();
        scores_agg=scores_minimal.groupby(['subject id','calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_agg=scores_agg.round({'calib_level_wholegests':5})
        scores_agg.pivot(index='calib_level_wholegests',
                         columns='subject id',
                         values='mean').plot(kind='bar',ax=ax,rot=0)#,capsize=2,width=0.8,
                                            # yerr=scores_agg.pivot(index='calib_level_wholegests',
                                            #                       columns='subject id',values='std'))
        '''only relevant yerr here would be if i got multiple shots per ppt - which would be nice'''
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Accuracy per subject on reserved 33% of session 3 (66 gests)')
        ax.set_xlabel('# Session 3 gestures used (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        plt.axhline(y=0.7475,label='Train 1&2\n(no cal) avg',linestyle='--',color='black')
        ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        plt.show()
        
        
        fig,ax=plt.subplots();
        scores_agg=scores_minimal.groupby(['subject id','calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_agg=scores_agg.round({'calib_level_wholegests':5})
        scores_agg.pivot(index='calib_level_wholegests',
                         columns='subject id',
                         values='mean').plot(kind='line',ax=ax,rot=0)#,capsize=2,width=0.8,
                                            # yerr=scores_agg.pivot(index='calib_level_wholegests',
                                            #                       columns='subject id',values='std'))
        '''only relevant yerr here would be if i got multiple shots per ppt - which would be nice'''
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Accuracy per subject on reserved 33% of session 3 (66 gests)')
        ax.set_xlabel('# Session 3 gestures used (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        plt.axhline(y=0.7475,label='Train 1&2\n(no cal) avg',linestyle='--',color='black')
        ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        plt.show()
        
        
        
        fig,ax=plt.subplots();
        scores_agg=scores_minimal.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_agg=scores_agg.round({'calib_level_wholegests':5})
        scores_agg.plot(y='mean',x='calib_level_wholegests',kind='bar',ax=ax,rot=0,yerr='std',capsize=5)
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies over subjects on reserved 33% of session 3 (66 gests)')
        ax.set_xlabel('# Session 3 gestures used (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        #plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        #plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        plt.axhline(y=0.7475,label='Train 1&2\n(no cal) avg',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
        plt.show()
        
        fig,ax=plt.subplots();
        scores_agg=scores_minimal.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_agg=scores_agg.round({'calib_level_wholegests':5})
        
        scores_calib_agg=scores_calib_minimal.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_calib_agg=scores_calib_agg.round({'calib_level_wholegests':5})
        
        scores_calib_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented\ntrained on 1&2')
        scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session3 only')
        ax.set_ylim(np.floor(scores_calib_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_calib_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies over subjects on reserved 33% of session 3 (66 gests)')
        ax.set_xlabel('# Session 3 gestures used (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        plt.axhline(y=0.86907,label='RQ2 Full Besp\n(Not session-split!)',linestyle='--',color='pink')
        #plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        plt.axhline(y=0.7475,label='Train 1&2\n(no cal) avg',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
        plt.show()

