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
from copy import deepcopy


def setup_warmstart_space(architecture='decision',include_svm=True):
    emgoptions=[
                {'emg_model_type':'RF',
                 'n_trees':scope.int(hp.quniform('emg.RF.ntrees',10,100,q=5)),
                 'max_depth':5,#scope.int(hp.quniform('emg.RF.maxdepth',2,5,q=1)),
                 #integerising search space https://github.com/hyperopt/hyperopt/issues/566#issuecomment-549510376
                 },
#                {'emg_model_type':'kNN',
#                 'knn_k':scope.int(hp.quniform('emg.knn.k',1,25,q=1)),
#                 },
#                {'emg_model_type':'LDA',
#                 'LDA_solver':hp.choice('emg.LDA_solver',['svd','lsqr','eigen']), #removed lsqr due to LinAlgError: SVD did not converge in linear least squares but readding as this has not repeated
#                 'shrinkage':hp.uniform('emg.lda.shrinkage',0.0,0.5),
#                 },
#                {'emg_model_type':'QDA', #emg qda reg 0.3979267 actually worked well!! for withinppt
#                 'regularisation':hp.uniform('emg.qda.regularisation',0.0,1.0), #https://www.kaggle.com/code/code1110/best-parameter-s-for-qda/notebook
#                 },
                {'emg_model_type':'LR',
                 'solver':hp.choice('emg.LR.solver',['sag']),
                 'C':hp.loguniform('emg.LR.c',np.log(0.01),np.log(10)),
                 },
                {'emg_model_type':'gaussNB',
                 'smoothing':hp.loguniform('emg.gnb.smoothing',np.log(1e-9),np.log(0.5e0)),
                 },
                ]
    eegoptions=[
                {'eeg_model_type':'RF',
                 'n_trees':scope.int(hp.quniform('eeg_ntrees',10,100,q=5)),
                 'max_depth':5,#scope.int(hp.quniform('eeg.RF.maxdepth',2,5,q=1)),
                 },
#                {'eeg_model_type':'LDA',
#                 'LDA_solver':hp.choice('eeg.LDA_solver',['svd','lsqr','eigen']),
#                 'shrinkage':hp.uniform('eeg.lda.shrinkage',0.0,0.5),
#                 },
#                {'eeg_model_type':'QDA',
#                 'regularisation':hp.uniform('eeg.qda.regularisation',0.0,1.0),
#                 },
                {'eeg_model_type':'LR',
                 'solver':hp.choice('eeg.LR.solver',['sag']),
                 'C':hp.loguniform('eeg.LR.c',np.log(0.01),np.log(10)),
                 },
                {'eeg_model_type':'gaussNB',
                 'smoothing':hp.loguniform('eeg.gnb.smoothing',np.log(1e-9),np.log(1e0)),
                 },
                ]
 #   if include_svm:
 #       emgoptions.append({'emg_model_type':'SVM_PlattScale',
 #                'kernel':hp.choice('emg.svm.kernel',['rbf']),#'poly','linear']),
 #                'svm_C':hp.loguniform('emg.svm.c',np.log(0.1),np.log(100)), #use loguniform? #https://queirozf.com/entries/choosing-c-hyperparameter-for-svm-classifiers-examples-with-scikit-learn
 #                'gamma':hp.loguniform('emg.svm.gamma',np.log(0.01),np.log(0.2)), #maybe log, from lower? #https://vitalflux.com/svm-rbf-kernel-parameters-code-sample/
 #                #eg sklearns gridsearch doc uses SVC as an example with C log(1e0,1e3) & gamma log(1e-4,1e-3)
 #                })
 #       eegoptions.append({'eeg_model_type':'SVM_PlattScale',
 #                'kernel':hp.choice('eeg.svm.kernel',['rbf']),#'poly','linear']),
 #                'svm_C':hp.loguniform('eeg.svm.c',np.log(0.01),np.log(100)),
 #                'gamma':hp.loguniform('eeg.svm.gamma',np.log(0.01),np.log(0.2)), 
 #                #https://www.kaggle.com/code/donkeys/exploring-hyperopt-parameter-tuning?scriptVersionId=12655875&cellId=64
 #                # naming convention https://github.com/hyperopt/hyperopt/issues/380#issuecomment-685173200
 #                })
    space = {
            'emg':hp.choice('emg model',emgoptions),
            'eeg':hp.choice('eeg model',eegoptions),
            'fusion_alg':hp.choice('fusion algorithm',[
                {'fusion_alg_type':'mean'},
                {'fusion_alg_type':'3_1_emg'},
                {'fusion_alg_type':'highest_conf'},
#                {'fusion_alg_type':'svm',
#                 'svm_C':hp.loguniform('fus.svm.c',np.log(0.01),np.log(100)),
#                 },
#                {'fusion_alg_type':'lda',
#                 'LDA_solver':hp.choice('fus.LDA.solver',['svd','lsqr','eigen']),
#                 'shrinkage':hp.uniform('fus.lda.shrinkage',0.0,1.0),
#                 },
                {'fusion_alg_type':'rf',
                 'n_trees':scope.int(hp.quniform('fus.RF.ntrees',10,100,q=5)),
                 'max_depth':5,
                 },
                ]),
            #'emg_set_path':params.jeong_EMGfeats,
            'emg_set_path':params.jeong_emg_noholdout,
            #'eeg_set_path':params.jeong_noCSP_WidebandFeats,
            'eeg_set_path':params.jeong_eeg_noholdout,
            #'eeg_set_path':'H:/Jeong11tasks_data/DUMMY_EEG.csv',
            'using_literature_data':True,
            'data_in_memory':False,
            'prebalanced':False,
            'scalingtype':'standardise',
            #'scalingtype':hp.choice('scaling',['normalise','standardise']),#,None]),
            'plot_confmats':False,
            'get_train_acc':False,
            'bag_eeg':False,
            'stack_distros':True,#hp.choice('decision.stack_distros',[True,False]),
            }
        
    return space

def warm_model(model,calib_data):
    train=calib_data.values[:,:-1]
    targets=calib_data.values[:,-1]
    model.set_params(warm_start=True)
    model.fit(train.astype(np.float64),targets)
    return model

def partfit_model(model,calib_data):
    train=calib_data.values[:,:-1]
    targets=calib_data.values[:,-1]
    model.partial_fit(train.astype(np.float64),targets)
    return model

def warm_cal_models(emg_model,eeg_model,emg_calib,eeg_calib,args):
    if emg_calib.empty:
        return emg_model,eeg_model
    
    if args['emg']['emg_model_type']=='gaussNB':
        emg_model=partfit_model(emg_model,emg_calib)
    else:
        emg_model=warm_model(emg_model,emg_calib)
    if args['eeg']['eeg_model_type']=='gaussNB':
        eeg_model=partfit_model(eeg_model,eeg_calib)
    else:
        eeg_model=warm_model(eeg_model,eeg_calib)
    return emg_model,eeg_model

def warm_cal_fuser(fuser, mode1, mode2, fustargets, args):
    train=np.column_stack([mode1,mode2])
    fuser.set_params(warm_start=True)
    fuser.fit(train.astype(np.float64),fustargets)
    return fuser

def stack_fusion(fuser,onehot,predlist_emg,predlist_eeg,classlabels):
    if onehot is not None:
        predlist_emg=fusion.encode_preds_onehot(predlist_emg,onehot)
        predlist_eeg=fusion.encode_preds_onehot(predlist_eeg,onehot)
    fusion_preds=fuser.predict(np.column_stack([predlist_emg,predlist_eeg]))
    return fusion_preds

def fusion_stack_warm(emg_train, eeg_train, emg_calib, eeg_calib, emg_test, eeg_test, args):
    emg_model,eeg_model,fuser,onehotEncoder=train_fuse_stack(emg_train, eeg_train, args,heat='cold')
    if not emg_calib.empty:
        emg_model,eeg_model,fuser,onehotEncoder=train_fuse_stack(emg_calib, eeg_calib, args,emg_model,eeg_model,fuser,heat='warm')
    classlabels=emg_model.classes_
    
    emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
    targets, predlist_emg, predlist_eeg, _, distros_emg, distros_eeg, _  = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
    
    predlist_fusion=stack_fusion(fuser,onehotEncoder,distros_emg,distros_eeg,classlabels)
    
    return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels

def fusion_stack_cold(emg_train, eeg_train, emg_test, eeg_test, args):
    emg_model,eeg_model,fuser,onehotEncoder=train_fuse_stack(emg_train, eeg_train, args,heat='cold')
    classlabels=emg_model.classes_
    
    emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
    targets, predlist_emg, predlist_eeg, _, distros_emg, distros_eeg, _  = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
    
    predlist_fusion=stack_fusion(fuser,onehotEncoder,distros_emg,distros_eeg,classlabels)
    
    return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels

def train_fuse_stack(emg_train, eeg_train, args, emg_model=None,eeg_model=None,fuser=None, heat='cold'):
    emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_train=emg_train.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_train=eeg_train.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
            
    emg_train['ID_splitIndex']=emg_train['Label'].astype(str)+emg_train['ID_pptID'].astype(str)
    eeg_train['ID_splitIndex']=eeg_train['Label'].astype(str)+eeg_train['ID_pptID'].astype(str)
    #https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
    
    sel_cols_eeg=args['sel_cols_eeg']
    sel_cols_emg=args['sel_cols_emg']
    
    random_split=random.randint(0,100)
    folds=StratifiedKFold(random_state=random_split,n_splits=3,shuffle=False)
    fustargets=[]
    fusdistros_emg=[]
    fusdistros_eeg=[]
    for i, (index_ML, index_Fus) in enumerate(folds.split(emg_train,emg_train['ID_splitIndex'])):
        emg_train_split_ML=emg_train.iloc[index_ML]
        emg_train_split_fusion=emg_train.iloc[index_Fus]
        eeg_train_split_ML=eeg_train.iloc[index_ML]
        eeg_train_split_fusion=eeg_train.iloc[index_Fus]
        
        emg_train_split_ML=ml.drop_ID_cols(emg_train_split_ML)
        eeg_train_split_ML=ml.drop_ID_cols(eeg_train_split_ML)
        
        emg_train_split_ML=emg_train_split_ML.iloc[:,sel_cols_emg]
        eeg_train_split_ML=eeg_train_split_ML.iloc[:,sel_cols_eeg]
        
        if heat=='cold':
            emg_modelTemp,eeg_modelTemp=train_models_opt(emg_train_split_ML,eeg_train_split_ML,args)
            classlabels = emg_modelTemp.classes_
        elif heat=='warm':
            emg_modelTemp,eeg_modelTemp=warm_cal_models(deepcopy(emg_model),deepcopy(eeg_model),emg_train_split_ML,eeg_train_split_ML,args)
            classlabels = emg_modelTemp.classes_
        
        targets,predlist_emg,predlist_eeg,_,distros_emg,distros_eeg,_=refactor_synced_predict(emg_train_split_fusion, eeg_train_split_fusion, emg_modelTemp, eeg_modelTemp, classlabels, args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
        if len(fustargets)==0:
            fustargets=targets
            fusdistros_emg=distros_emg
            fusdistros_eeg=distros_eeg
        else:
            fustargets=fustargets+targets
            fusdistros_emg=np.concatenate((fusdistros_emg,distros_emg),axis=0)
            fusdistros_eeg=np.concatenate((fusdistros_eeg,distros_eeg),axis=0)
            
    emg_train=ml.drop_ID_cols(emg_train)
    emg_train=emg_train.iloc[:,sel_cols_emg]
    eeg_train=ml.drop_ID_cols(eeg_train)
    eeg_train=eeg_train.iloc[:,sel_cols_eeg]
    onehotEncoder=None
    
    if heat=='cold':
        emg_model,eeg_model=train_models_opt(emg_train,eeg_train,args)
        if args['fusion_alg']['fusion_alg_type']=='svm':
            fuser=fusion.train_svm_fuser(fusdistros_emg, fusdistros_eeg, fustargets, args['fusion_alg'])
        elif args['fusion_alg']['fusion_alg_type']=='lda':
            fuser=fusion.train_lda_fuser(fusdistros_emg, fusdistros_eeg, fustargets, args['fusion_alg'])
        elif args['fusion_alg']['fusion_alg_type']=='rf':
            fuser=fusion.train_rf_fuser(fusdistros_emg, fusdistros_eeg, fustargets, args['fusion_alg'])
    elif heat=='warm':
        emg_model,eeg_model=warm_cal_models(emg_model,eeg_model,emg_train,eeg_train,args)
        fuser=warm_cal_fuser(fuser, fusdistros_emg, fusdistros_eeg, fustargets, args)
    
    return emg_model,eeg_model,fuser,onehotEncoder

def fuse_opt_coldstart(args):
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
   
    eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(train_split[0])]
    emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(train_split[0])]
    
    eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split[0])]
    emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]
 
    if len(emg_test['Label'].value_counts())<4:
        raise ValueError('Not all gestures in optimisation-test split')
        
    sel_cols_emg=args['sel_cols_emg']
    sel_cols_eeg=args['sel_cols_eeg'] 
        
    if args['fusion_alg']['fusion_alg_type']=='svm':      
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_stack_cold(emg_train, eeg_train, emg_test, eeg_test, args)
    
    elif args['fusion_alg']['fusion_alg_type']=='lda':           
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_stack_cold(emg_train, eeg_train, emg_test, eeg_test, args)
    
    elif args['fusion_alg']['fusion_alg_type']=='rf':    
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_stack_cold(emg_train, eeg_train, emg_test, eeg_test, args)
    
    else:        

        emg_train=ml.drop_ID_cols(emg_train)
        eeg_train=ml.drop_ID_cols(eeg_train)
        
        eeg_train=eeg_train.iloc[:,sel_cols_eeg]
        emg_train=emg_train.iloc[:,sel_cols_emg]
        
        emg_model,eeg_model=train_models_opt(emg_train,eeg_train,args)
    
        classlabels = emg_model.classes_
        
        emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            
        targets, predlist_emg, predlist_eeg, predlist_fusion,_,_,_ = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args, sel_cols_eeg,sel_cols_emg)


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

def fuse_warmstart(args):
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
   
    if args['opt_method']=='no calib' or args['calib_level']==0:
        eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(train_split[0])]
        emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(train_split[0])]
        
        eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split[0])]
        emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]
    
    
    elif args['opt_method']=='calib': #'cross_session_cal'
 
        eeg_train=eeg_ppt[eeg_ppt['ID_run']!=3.0][eeg_ppt['ID_stratID'].isin(train_split[0])]
        emg_train=emg_ppt[emg_ppt['ID_run']!=3.0][emg_ppt['ID_stratID'].isin(train_split[0])]
        
        if args['calib_level']==4/134:
            raise ValueError('Insufficient calib data to warmstart during opt')
                      
        else:
            emg_calib=emg_ppt[emg_ppt['ID_run']==3.0]
            eeg_calib=eeg_ppt[eeg_ppt['ID_run']==3.0]
            
            gest_perfs=emg_calib['ID_stratID'].unique()
            gest_strat=pd.DataFrame([gest_perfs,[perf.split('.')[1][-1] for perf in gest_perfs]]).transpose()
            
            random_split=random.randint(0,100)
                        
            if args['calib_level']==8/134:

                traincal_split,test_split=train_test_split(gest_strat,test_size=0.5,
                                                        random_state=random_split,stratify=gest_strat[1])
            elif args['calib_level'] > 9/134:

                traincal_split,test_split=train_test_split(gest_strat,test_size=args['testset_size'],
                                                        random_state=random_split,stratify=gest_strat[1])    
                
            emg_test=emg_calib[emg_calib['ID_stratID'].isin(test_split[0])]
            eeg_test=eeg_calib[eeg_calib['ID_stratID'].isin(test_split[0])]
            emg_calib=emg_calib[emg_calib['ID_stratID'].isin(traincal_split[0])]
            eeg_calib=eeg_calib[eeg_calib['ID_stratID'].isin(traincal_split[0])]
           # emg_train=pd.concat([emg_train,emg_calib])
           # eeg_train=pd.concat([eeg_train,eeg_calib])
            
            # haha wait, shouldnt be appending them for warm start...

    
    '''where we use calib data to opt, we optimise for the calib data within that 1/3rd (of opt data)'''
    '''so we dont make use of all data in every opt loop, but were still using it overall'''
    '''in cases where there is such little calib data as to be unable to split:'''
    '''  -- for NOW we are either optimising for all the session 3 data (ie all 1 repeat of each gesture)'''
    ''' -- or splitting them some other way, so if 2 repeats, using 1 for within-opt train and one for target'''
    ''' -- (adding that to the 2/3rds of non-session data used within the opt loop)'''
    ''' -- we COULD see whether such a tiny amount of calib is better used for opt / train / opt-train / opt-targ'''

    if len(emg_test['Label'].value_counts())<4:
        raise ValueError('Not all gestures in optimisation-test split')
    
    
    sel_cols_emg=args['sel_cols_emg']
    sel_cols_eeg=args['sel_cols_eeg'] 
        
    if args['fusion_alg']['fusion_alg_type']=='svm':      
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_stack_warm(emg_train, eeg_train, emg_calib, eeg_calib, emg_test, eeg_test, args)
    
    elif args['fusion_alg']['fusion_alg_type']=='lda':           
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_stack_warm(emg_train, eeg_train, emg_calib, eeg_calib, emg_test, eeg_test, args)
    
    elif args['fusion_alg']['fusion_alg_type']=='rf':    
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_stack_warm(emg_train, eeg_train, emg_calib, eeg_calib, emg_test, eeg_test, args)
    
    else:        

        emg_train=ml.drop_ID_cols(emg_train)
        eeg_train=ml.drop_ID_cols(eeg_train)
        
        eeg_train=eeg_train.iloc[:,sel_cols_eeg]
        emg_train=emg_train.iloc[:,sel_cols_emg]
        
        emg_model,eeg_model=train_models_opt(emg_train,eeg_train,args)
        
        emg_calib=ml.drop_ID_cols(emg_calib)
        eeg_calib=ml.drop_ID_cols(eeg_calib)
        
        eeg_calib=eeg_calib.iloc[:,sel_cols_eeg]
        emg_calib=emg_calib.iloc[:,sel_cols_emg]
        
        emg_model,eeg_model=warm_cal_models(emg_model,eeg_model,emg_calib,eeg_calib,args)
    
        classlabels = emg_model.classes_
        
        emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            
        targets, predlist_emg, predlist_eeg, predlist_fusion,_,_,_ = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args, sel_cols_eeg,sel_cols_emg)


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


def fusionWarm_test(emg_train,eeg_train,emg_test,eeg_test,args):
    start=time.time()
    
    eeg_calib=eeg_train[eeg_train['ID_run']==3.0]
    emg_calib=emg_train[emg_train['ID_run']==3.0]
    eeg_train=eeg_train[eeg_train['ID_run']!=3.0]
    emg_train=emg_train[emg_train['ID_run']!=3.0]
        
    if args['fusion_alg']['fusion_alg_type']=='svm':      
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_stack_warm(emg_train, eeg_train, emg_calib, eeg_calib, emg_test, eeg_test, args)
    
    elif args['fusion_alg']['fusion_alg_type']=='lda':           
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_stack_warm(emg_train, eeg_train, emg_calib, eeg_calib, emg_test, eeg_test, args)
    
    elif args['fusion_alg']['fusion_alg_type']=='rf':    
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_stack_warm(emg_train, eeg_train, emg_calib, eeg_calib, emg_test, eeg_test, args)
    
    else:        
        emg_train=ml.drop_ID_cols(emg_train)
        eeg_train=ml.drop_ID_cols(eeg_train)
        
        eeg_train=eeg_train.iloc[:,sel_cols_eeg]
        emg_train=emg_train.iloc[:,sel_cols_emg]
        
        emg_model,eeg_model=train_models_opt(emg_train,eeg_train,args)
        
        emg_calib=ml.drop_ID_cols(emg_calib)
        eeg_calib=ml.drop_ID_cols(eeg_calib)
        
        eeg_calib=eeg_calib.iloc[:,sel_cols_eeg]
        emg_calib=emg_calib.iloc[:,sel_cols_emg]
        
        emg_model,eeg_model=warm_cal_models(emg_model,eeg_model,emg_calib,eeg_calib,args)
    
        classlabels = emg_model.classes_
        
        emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            
        targets, predlist_emg, predlist_eeg, predlist_fusion,_,_,_ = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args, sel_cols_eeg,sel_cols_emg)
    
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
    
    eeg_others['ID_stratID'] = eeg_others['ID_run'].astype(str)+eeg_others['Label'].astype(str)+eeg_others['ID_pptID'].astype(str)+eeg_others['ID_gestrep'].astype(str)
    emg_others['ID_stratID'] = emg_others['ID_run'].astype(str)+emg_others['Label'].astype(str)+emg_others['ID_pptID'].astype(str)+emg_others['ID_gestrep'].astype(str)
    random_split=random.randint(0,100)
    
    if not emg_others['ID_stratID'].equals(eeg_others['ID_stratID']):
        raise ValueError('EMG & EEG performances misaligned')
    gest_perfs=emg_others['ID_stratID'].unique()
    gest_strat=pd.DataFrame([gest_perfs,[(perf.split('.')[1])+(perf.split('.')[2]) for perf in gest_perfs]]).transpose()
    
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
    #load_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\D1_Warmstart_final_resMinimal - Copy.csv"
    
    #this is <warmstart from 1+2 train, no further opt>
    # ie Xfer No Opt
   # load_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\D2_Warmstart_NoOpt_final_resMinimal - Copy.csv"
    #RF was broken, fixed is below
    load_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\D2a_WarmNoOpt_FixRF_final_resMinimal - Copy.csv"
    
    #load_xfer_opt_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\D1_1_Warmstart_NoAppend_final_resMinimal - Copy.csv"
    #RF was broken, fixed is below
    load_xfer_opt_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\D1a_Warmstart_FixRF_final_resMinimal - Copy.csv"
    
    load_sessiononly_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\C1_Session3onlyfinal_resMinimal - Copy.csv"
    load_aug_unadjusted=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\B1_AugPipelinefinal_resMinimal - Copy.csv"
    
    load_aug_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\B1_1_AugAdjustedSplit_final_resMinimal - Copy.csv"
    
    load_withinSesh_noOpt=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\B2_WithinSession_noCal_final_resMinimal.csv"
    
    #load_warm_from_Gen=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\E2_Warmstart_fromGen_samesize_final_resMinimal.csv"
    #RF was broken, fixed is below
    load_warm_from_Gen=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\E2a_WarmFromGen_FixRF_final_resMinimal.csv"
    
    
    
    within_opt_2_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\B2a_WithinSession_optFor2_final_resMinimal.csv"
    within_opt_both_downsample_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\B2b_WithinSession_optForHalf_final_resMinimal.csv"
    train_both_baseline_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\A2_both1and2final_resMinimal.csv"
    train_1_baseline_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\A1a_session1final_resMinimal.csv"
    train_2_baseline_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\A1b_session2final_resMinimal.csv"
    train_both_downsample_baseline_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\A2a_bothNoExtraData_final_resMinimal.csv"
    withinTrain_topupOpt_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\C2_WithinSession_TopupOptfinal_resMinimal - Copy.csv"
    
    
    
    
    systemUnderTest = 'D2_Warmstart_NoOpt'
    
    testset_size = 0.33
        
    if systemUnderTest == 'D2_Warmstart_NoOpt':
        
        calib_levels = [0/134,4/134,8/134,20/134,40/134,60/134,72/134,80/134,100/134,120/134,132/134]
        # can't do 4/134 as not left with enough calib data to warmstart and use for opt target
                
        feats_method='no calib'
        opt_method='no calib'
        train_method='calib'
        
        train_session='both'
        
        calib_levels=[0]
                
        calib_levels = np.array([round(scale/(4/134))*(4/134) for scale in calib_levels])
        
    n_repeats = 1   
    
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
        
    
        for idx,emg_mask in enumerate(emg_masks):
                space=setup_warmstart_space(architecture='decision',include_svm=True)
                
                space.update({'l1_sparsity':0.05})
                space.update({'l1_maxfeats':40})
                
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
                
                eeg_ppt['ID_stratID']=eeg_ppt['ID_run'].astype(str)+eeg_ppt['Label'].astype(str)+emg_ppt['ID_pptID'].astype(str)+eeg_ppt['ID_gestrep'].astype(str)
                emg_ppt['ID_stratID']=emg_ppt['ID_run'].astype(str)+emg_ppt['Label'].astype(str)+emg_ppt['ID_pptID'].astype(str)+emg_ppt['ID_gestrep'].astype(str)
                random_split=random.randint(0,100)
                
                if not emg_ppt['ID_stratID'].equals(eeg_ppt['ID_stratID']):
                    raise ValueError('EMG & EEG performances misaligned')
                
                
                for repeat in range(n_repeats):
                    
                    emg_session1=emg_ppt[emg_ppt['ID_run']==1.0]
                    eeg_session1=eeg_ppt[eeg_ppt['ID_run']==1.0]
                    emg_session2=emg_ppt[emg_ppt['ID_run']==2.0]
                    eeg_session2=eeg_ppt[eeg_ppt['ID_run']==2.0]
                    emg_session3=emg_ppt[emg_ppt['ID_run']==3.0]
                    eeg_session3=eeg_ppt[eeg_ppt['ID_run']==3.0]
                                                      
                    if train_session=='first':
                        emg_train=emg_session1
                        eeg_train=eeg_session1
                    elif train_session=='second':
                        emg_train=emg_session2
                        eeg_train=eeg_session2
                    elif train_session=='both':
                        emg_train=pd.concat([emg_session1,emg_session2])
                        eeg_train=pd.concat([eeg_session1,eeg_session2])
                    
                    trials=Trials()
                    
                    emg_train,emgscaler=feats.scale_feats_train(emg_train,space['scalingtype'])
                    eeg_train,eegscaler=feats.scale_feats_train(eeg_train,space['scalingtype'])
                    
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
                    
                    best = fmin(fuse_opt_coldstart,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=iters,
                                trials=trials)
                        
                    winner_args=space_eval(space,best)
                    best_loss=trials.best_trial['result']['loss']
                    '''BELOW IS TEMP ONLY'''
                    #winner_args=adjust_space(space,space['subject-id'])
                    #best_loss=0
                    '''***'''
            
                    winner_args['sel_cols_emg']=sel_cols_emg
                    winner_args['sel_cols_eeg']=sel_cols_eeg 
                    winner_args['subject id']=str(int(eeg_ppt['ID_pptID'][0]))
                    
                    if n_repeats > 1:
                        winner_args['plot_confmats']=False
                    else:
                        winner_args['plot_confmats']=True
                
                    
                    
                    for calib_level in calib_levels:
                        print('Calib level: ',str(calib_level),' (subject ',str(idx),' of 20)')
                        
                        winner_args.update({'calib_level':calib_level})
                        '''here we are NOT doing any cal in the opt. if we want to opt a system which uses some
                        level of cal on itself, then we will have to use the line below & do a new opt for each
                        possible level of cal'''
                        #space.update({'calib_level':calib_level})
                        
                        gest_perfs=emg_session3['ID_stratID'].unique()
                        gest_strat=pd.DataFrame([gest_perfs,[perf.split('.')[1][-1] for perf in gest_perfs]]).transpose()
                        
                        calib_split,test_split=train_test_split(gest_strat,test_size=testset_size,
                                                              random_state=random_split,stratify=gest_strat[1])
                            
                        emg_test=emg_session3[emg_session3['ID_stratID'].isin(test_split[0])]
                        eeg_test=eeg_session3[eeg_session3['ID_stratID'].isin(test_split[0])]                            
                        
                        emg_test=feats.scale_feats_test(emg_test,emgscaler)
                        eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
                        
                        if calib_level==0:
                            subject_results=fusionWarm_test(emg_train,eeg_train,emg_test,eeg_test,winner_args)
                            subject_results['best_loss']=best_loss
                            subject_results['repeat']=repeat
                            
                            ppt_winners.append(winner_args.copy())
                            ppt_results.append(subject_results)
                            continue
                        
                        if calib_level > 130/134:
                            '''case for when we cant train_test_split, as the (unused) "test" would be < n_classes'''
                            stratsize=np.min(calib_split[1].value_counts())
                            calib_split = calib_split.groupby(1,axis=0)
                            calib_split=calib_split.apply(lambda x: x.sample(stratsize))
                        elif 0 < calib_level < 1:
                            calib_split,_=train_test_split(calib_split,train_size=calib_level,random_state=random_split,stratify=calib_split[1])
                            if len(calib_split[1].value_counts()) < 4:
                                print('Not all gestures present in calib of'+str(calib_level))
                                skipRolloff=True
                                break
                            if 0:
                                if min(calib_split[1].value_counts()) < 2:
                                    print('calib of ' +str(calib_level)+' results in < 2 performances per class')
                                    if calib_level==4/134:
                                        print('special exception where 1 gesture per class is used for calibration')
                                    else:
                                        skipRolloff=True
                                        break
                            
                        emg_calib=emg_session3[emg_session3['ID_stratID'].isin(calib_split[0])]
                        eeg_calib=eeg_session3[eeg_session3['ID_stratID'].isin(calib_split[0])]
                        
                        emg_calib=feats.scale_feats_test(emg_calib,emgscaler)
                        eeg_calib=feats.scale_feats_test(eeg_calib,eegscaler)

                        emg_joint = pd.concat([emg_train,emg_calib])
                        eeg_joint = pd.concat([eeg_train,eeg_calib])
                             
                        subject_results=fusionWarm_test(emg_joint,eeg_joint,emg_test,eeg_test,winner_args)
                        subject_results['best_loss']=best_loss
                        subject_results['repeat']=repeat
                        
                        ppt_winners.append(winner_args.copy())
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
                scores_minimal=results[['subject id','fusion_acc','emg_acc','eeg_acc','elapsed_time','repeat',
                                'train_session','fusion_alg_fusion_alg_type','eeg_eeg_model_type','emg_emg_model_type',
                               'featsel_method','opt_method','train_method','calib_level','best_loss','opt_acc']]
                
        currentpath=os.path.dirname(__file__)
        result_dir=params.jeong_results_dir
        resultpath=os.path.join(currentpath,result_dir)    
        resultpath=os.path.join(resultpath,'RQ3')
       
        #calib_label='_calib'+str(round(calib_level,3))
        
        #picklepath=os.path.join(resultpath,(systemUnderTest+calib_label+'_resDF.pkl'))
        #csvpath=os.path.join(resultpath,(systemUnderTest+calib_label+'_resMinimal.csv'))

       # pickle.dump(results,open(picklepath,'wb'))
        #scores_minimal.to_csv(csvpath)
        
        picklefullpath=os.path.join(resultpath,(systemUnderTest+'_final_resDF.pkl'))
        csvfullpath=os.path.join(resultpath,(systemUnderTest+'_final_resMinimal.csv'))

        pickle.dump(results,open(picklefullpath,'wb'))
        scores_minimal.to_csv(csvfullpath)
    else:
        scores_minimal=pd.read_csv(load_res_path,index_col=0)        
    
    if plot_results:
        scores_aug_minimal=pd.read_csv(load_aug_res_path,index_col=0)
        scores_sessiononly_minimal=pd.read_csv(load_sessiononly_res_path,index_col=0)
        scores_xfer_opt=pd.read_csv(load_xfer_opt_path,index_col=0)
        scores_session_noOpt=pd.read_csv(load_withinSesh_noOpt,index_col=0)
        scores_XferGen=pd.read_csv(load_warm_from_Gen,index_col=0)
        
       # scores_aug_unadjusted=pd.read_csv(load_aug_unadjusted,index_col=0)
        
        
        scores_minimal=scores_minimal.round({'augment_scale':5})
        
        plt.rcParams['figure.dpi'] = 150 # DEFAULT IS 100
           
        
        nGest=4
        nRepsPerGest=50
        nInstancePerGest=4
        trainsplitSize=2/3
        scores_minimal['calib_level_instances']=scores_minimal['calib_level']*(1-testset_size)*nGest*nRepsPerGest*nInstancePerGest
        scores_minimal['calib_level_wholegests']=scores_minimal['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        scores_minimal['calib_level_pergest']=scores_minimal['calib_level']*(1-testset_size)*nRepsPerGest
        
                
        scores_aug_minimal['calib_level_wholegests']=scores_aug_minimal['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        scores_sessiononly_minimal['calib_level_wholegests']=scores_sessiononly_minimal['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        scores_xfer_opt['calib_level_wholegests']=scores_xfer_opt['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        scores_session_noOpt['calib_level_wholegests']=scores_session_noOpt['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        scores_XferGen['calib_level_wholegests']=scores_XferGen['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        
        
        rq2_bespoke_ref_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ2\D1e_NoRolloff_Stablefinal_resMinimal - Copy.csv"
        scores_rq2=pd.read_csv(rq2_bespoke_ref_path,index_col=0)  
        scores_rq2['augscale_wholegests']=np.around(scores_rq2['augment_scale']*19*nGest*nRepsPerGest).astype(int)
        scores_rq2['trainAmnt_wholegests']=np.around(scores_rq2['rolloff_factor']*trainsplitSize*nGest*nRepsPerGest).astype(int)
        
        noAugRQ2=scores_rq2[scores_rq2['augscale_wholegests']==0]
        noAugRQ2=noAugRQ2.groupby(['trainAmnt_wholegests','augscale_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        
        

        
        if 0:
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
            ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
            ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
            
            plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
            plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
            plt.axhline(y=0.7475,label='Train both\n(no cal) avg',linestyle='--',color='black')
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
        ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        #plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
#        plt.axhline(y=noAugRQ2['mean'][0],label='RQ2 Bespoke\n(Not session-split)',linestyle='--',color='pink')
#        plt.axhline(y=noAugRQ2['mean'][0]+noAugRQ2['std'][0],linestyle=':',color='pink')
#        plt.axhline(y=noAugRQ2['mean'][0]-noAugRQ2['std'][0],linestyle=':',color='pink')
        
        plt.axhline(y=0.723,label='Proxy* Generalist',linestyle='--',color='gray')
        #plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        plt.axhline(y=0.7475,label='Train both\n(no cal) avg',linestyle='--',color='black')
        ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        plt.show()
        
        
        '''
        fig,ax=plt.subplots();
        scores_agg=scores_minimal.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_agg=scores_agg.round({'calib_level_wholegests':5})
        scores_agg.plot(y='mean',x='calib_level_wholegests',kind='bar',ax=ax,rot=0,yerr='std',capsize=5)
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies over subjects on reserved 33% of session 3 (66 gests)')
        ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        #plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        #plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        plt.axhline(y=0.7475,label='Train both\n(no cal) avg',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
        plt.show()
        '''
        
        
        
        
        
        scores_agg=scores_minimal.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_agg=scores_agg.round({'calib_level_wholegests':5})
        
        scores_xfer_opt_agg=scores_xfer_opt.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_xfer_opt_agg=scores_xfer_opt_agg.round({'calib_level_wholegests':5})
        
        scores_aug_agg=scores_aug_minimal.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_aug_agg=scores_aug_agg.round({'calib_level_wholegests':5})
        
        scores_sessiononly_agg=scores_sessiononly_minimal.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_sessiononly_agg=scores_sessiononly_agg.round({'calib_level_wholegests':5})
        
        scores_sessionNoOpt_agg=scores_session_noOpt.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_sessionNoOpt_agg=scores_sessionNoOpt_agg.round({'calib_level_wholegests':5})
        
        scores_xfer_gen_agg=scores_XferGen.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_xfer_gen_agg=scores_xfer_gen_agg.round({'calib_level_wholegests':5})
        
        
        fig,ax=plt.subplots();
        
        
        if 0:
            scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only',yerr='std',capsize=5)
            scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented',yerr='std',capsize=5)
            scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\ntrained on 1&2',yerr='std',capsize=5)
            scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nwithout opt',yerr='std',capsize=5)
            scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only\nwithout opt',yerr='std',capsize=5)
            scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist',yerr='std',capsize=5)
        else:
            scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only')
            scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented')
            scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\ntrained on 1&2')
            scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nwithout opt')
            scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only\nwithout opt')
            scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist')
                    
 #       scores_aug_unadjusted['calib_level_wholegests']=scores_aug_unadjusted['calib_level']*(1-testset_size)*nGest*nRepsPerGest
 #       aug_unadjust_agg=scores_aug_adjusted.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
 #       aug_unadjust_agg=aug_adjust_agg.round({'calib_level_wholegests':5})
 #       aug_unadjust_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented\nadjusted split')
                
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies over subjects on reserved 33% of session 3 (66 gests)')
        ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        #below is 869066 taken from RQ2's A1_FullBespoke (in "provis")
        # but MIGHT actually be 865025 if we used D1b_RolloffStable5trials (which we use for RQ2 Hyp1)
        # OR 866105 if we use D1a_AugStable_with_errorCalcs
      #  plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        # NOW we take it from RQ2 rolloffStable5trials, but corrected (previously 5 trials werent different)
#        plt.axhline(y=noAugRQ2['mean'][0],label='RQ2 Bespoke\n(Not session-split)',linestyle='--',color='pink')
#        plt.axhline(y=noAugRQ2['mean'][0]+noAugRQ2['std'][0],linestyle=':',color='pink')
#        plt.axhline(y=noAugRQ2['mean'][0]-noAugRQ2['std'][0],linestyle=':',color='pink')
        
        #plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        #ax.set_ylim(0.4,0.9)
        plt.axhline(y=0.7475,label='Train 1&2\n(no cal) avg',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
        plt.show()
        
        
        
        
        '''** ALL **'''
        
        fig,ax=plt.subplots();
        
        within_opt_2=pd.read_csv(within_opt_2_path,index_col=0)
        within_opt_both_downsample=pd.read_csv(within_opt_both_downsample_path,index_col=0)
        train_both_baseline=pd.read_csv(train_both_baseline_path,index_col=0)
        train_2_baseline=pd.read_csv(train_2_baseline_path,index_col=0)
        train_1_baseline=pd.read_csv(train_1_baseline_path,index_col=0)
        train_both_downsample_baseline=pd.read_csv(train_both_downsample_baseline_path,index_col=0)
        topupOpt=pd.read_csv(withinTrain_topupOpt_path,index_col=0)
        
        
        within_opt_2['calib_level_wholegests']=within_opt_2['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        within_opt_both_downsample['calib_level_wholegests']=within_opt_both_downsample['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        train_both_baseline['calib_level_wholegests']=train_both_baseline['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        train_2_baseline['calib_level_wholegests']=train_2_baseline['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        train_1_baseline['calib_level_wholegests']=train_1_baseline['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        train_both_downsample_baseline['calib_level_wholegests']=train_both_downsample_baseline['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        topupOpt['calib_level_wholegests']=topupOpt['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        
        
        within_opt_2_agg=within_opt_2.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        within_opt_both_downsample_agg=within_opt_both_downsample.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        topupOpt_agg=topupOpt.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        
        train_both_baseline_score=np.mean(train_both_baseline['fusion_acc'])
        train_2_baseline_score=np.mean(train_2_baseline['fusion_acc'])
        train_1_baseline_score=np.mean(train_1_baseline['fusion_acc'])
        train_both_downsample_baseline_score=np.mean(train_both_downsample_baseline['fusion_acc'])
        
        
        if 0:
            scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only',yerr='std',capsize=5)
            scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented',yerr='std',capsize=5)
            scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\ntrained on 1&2',yerr='std',capsize=5)
            scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nwithout opt',yerr='std',capsize=5)
            scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only\nwithout opt',yerr='std',capsize=5)
            scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist',yerr='std',capsize=5)
        else:
            scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only')
            scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented')
            scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom 1 + 2 incl\nopt for transfer')
            scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom static 1 + 2\nno further opt')
            scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 train\nOptimised for 1+2')
            scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist')
            within_opt_2_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 train\nOptimised for 2')
            within_opt_both_downsample_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 train\nOptimised for 1+2\n(downsampled to half)')
            topupOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 train\nOptimised for 3 topped\nup by 1+2 to 200 total')
            plt.axhline(y=train_both_baseline_score,label='Train 1 + 2',linestyle='--',color='tab:orange')
            plt.axhline(y=train_2_baseline_score,label='Train session 2',linestyle='--',color='tab:green')
            plt.axhline(y=train_both_downsample_baseline_score,label='Train 1 + 2\n(downsampled to half)',linestyle='--',color='tab:red')
                    
 #       scores_aug_unadjusted['calib_level_wholegests']=scores_aug_unadjusted['calib_level']*(1-testset_size)*nGest*nRepsPerGest
 #       aug_unadjust_agg=scores_aug_adjusted.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
 #       aug_unadjust_agg=aug_adjust_agg.round({'calib_level_wholegests':5})
 #       aug_unadjust_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented\nadjusted split')
                
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('ALL Mean accuracies over Development subjects on reserved 33% of session 3 (66 gests)',loc='left')
        ax.set_xlabel('# Session 3 gestures (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        #below is 869066 taken from RQ2's A1_FullBespoke (in "provis")
        # but MIGHT actually be 865025 if we used D1b_RolloffStable5trials (which we use for RQ2 Hyp1)
        # OR 866105 if we use D1a_AugStable_with_errorCalcs
      #  plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        # NOW we take it from RQ2 rolloffStable5trials, but corrected (previously 5 trials werent different)
   #     plt.axhline(y=noAugRQ2['mean'][0],label='RQ2 Bespoke\n(Not session-split)',linestyle='--',color='pink')
   #     plt.axhline(y=noAugRQ2['mean'][0]+noAugRQ2['std'][0],linestyle=':',color='pink')
   #     plt.axhline(y=noAugRQ2['mean'][0]-noAugRQ2['std'][0],linestyle=':',color='pink')
        plt.axhline(y=0.723,label='Proxy* Generalist',linestyle='--',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        #ax.set_ylim(0.4,0.9)
       # plt.axhline(y=0.7475,label='Train 1&2\n(no cal) avg',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        plt.show()
        
        
        
        
        
        
        '''** JUST PRETRAINS **'''
        
      #  fig,ax=plt.subplots();
        fig=plt.figure()
        ax=fig.add_axes((0.0,0.15,0.8,0.8))
        
       # scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session learning')
       # scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Dataset augmented by\nall prior user data')
       # scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Transfer from prior user\ndata including optimisation')
       # scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Transfer from prior user\ndata with static configuration',c='tab:red')
       # scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Transfer from Generalist',c='tab:brown')
       # scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session training\n(system config optimised\non all prior user data)',c='tab:purple')
       # within_opt_2_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session training\n(system config optimised\non Session 2 user data)')
       # within_opt_both_downsample_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session training\n(system config optimised on\ndownsampled prior user data)',c='tab:gray')
       # topupOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session training\n(system config optimised on\njoint session-specific &\nprior user data)')
        plt.axhline(y=train_both_baseline_score,label='Optimised & pretrained \non all prior user data',linestyle='--',color='black')
        plt.axhline(y=train_1_baseline_score,label='Optimised & pretrained \non Session 1 data',linestyle='--',color='tab:blue')
        plt.axhline(y=train_2_baseline_score,label='Optimised & pretrained \non Session 2 data',linestyle='--',color='tab:orange')
        plt.axhline(y=train_both_downsample_baseline_score,label='Optimised & pretrained \non all prior user data\n(downsampled to half)',linestyle='--',color='tab:green')
                         
    #    ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies of pretrained systems over Development subjects\non reserved 33% of Session 3 data (66 gestures)',loc='left')
        ax.set_xlabel('# Session 3 gestures used for learning')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        

        plt.axhline(y=0.723,label='Proxy* Generalist',linestyle='-.',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        ax.set_ylim(0.4,1.0)
        ax.legend(loc='center left',bbox_to_anchor=(1,0.375),ncol=1)
        
        '''
        axTime=fig.add_axes((0.0,0.0,0.8,0.0))
        axTime.yaxis.set_visible(False)
        axTime.set_xticks(ax.get_xticks())
        def tick_function(X):
            #V = 1/(1+X)
            V=(X*3)/60
            return ["%.1f" % z for z in V]
        axTime.set_xticklabels(tick_function(ax.get_xticks()))
        axTime.set_xlim(ax.get_xlim())
        axTime.set_xlabel("Minimum session-specific recording time (minutes)")
        '''
        plt.show()
        
        
        
        fig,ax=plt.subplots()
        train_both_downsample_baseline_comp=train_both_downsample_baseline[train_both_downsample_baseline['repeat']==0].reset_index(drop=True)
        x=[train_both_baseline['fusion_acc'], train_1_baseline['fusion_acc'], train_2_baseline['fusion_acc'], train_both_downsample_baseline_comp['fusion_acc']]
        
        baselines=pd.concat(x,axis=1).set_axis(['Both prior sessions','Session 1','Session 2','Both sessions\n(downsampled to half)'],axis=1)
        basestack=baselines.stack().rename('fusion_acc')
        basestack=basestack.rename_axis(['devsubj','Baseline']).reset_index()
        basestack.boxplot(column='fusion_acc',by='Baseline',ax=ax,showmeans=True,positions=[0,3,1,2])
        #https://stackoverflow.com/questions/21508420/is-there-a-way-to-set-the-order-in-pandas-group-boxplots
        # default sort is alphabetical. could reverse map is but not worth effort
        #https://stackoverflow.com/questions/15541440/how-to-apply-custom-column-order-on-categorical-to-pandas-boxplot
        #https://stackoverflow.com/questions/70929624/pandas-get-first-occurrence-of-a-given-column-value
        
        ax.set_title('')
        ax.set_ylim(0.2,1)
        ax.set_xlabel('Prior subject data used to model baseline system')
        ax.set_ylabel('Accuracy')
        plt.suptitle('Accuracies of pretrained baseline system over Development\nsubjects on reserved 33% of Session 3 data (66 gestures)',y=1.0)
        plt.show()
        
        
        
        
        
        
        
        
        
        '''** ALL AFTER WHITTLING PRETRAINS **'''
        
      #  fig,ax=plt.subplots();
        fig=plt.figure()
        ax=fig.add_axes((0.0,0.15,0.8,0.8))
        
        scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session learning')
        scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Dataset augmented by\nall prior user data')
        scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Transfer from prior user\ndata including optimisation')
        scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Transfer from prior user\ndata with static configuration',c='tab:red')
        scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Transfer from Generalist',c='tab:brown')
        scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session training\n(system config optimised\non all prior user data)',c='tab:purple')
        within_opt_2_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session training\n(system config optimised\non Session 2 user data)')
        within_opt_both_downsample_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session training\n(system config optimised on\ndownsampled prior user data)',c='tab:gray')
        topupOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session training\n(system config optimised on\njoint session-specific &\nprior user data)')
        plt.axhline(y=train_both_baseline_score,label='Optimised & pretrained \non all prior user data',linestyle='--',color='black')
        #   plt.axhline(y=train_2_baseline_score,label='Train session 2',linestyle='--',color='tab:green')
        #   plt.axhline(y=train_both_downsample_baseline_score,label='Train 1 + 2\n(downsampled to half)',linestyle='--',color='tab:red')
                         
    #    ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies of all approaches over Development subjects\non reserved 33% of Session 3 data (66 gestures)',loc='left')
        ax.set_xlabel('# Session 3 gestures used for learning')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        

        plt.axhline(y=0.723,label='Proxy* Generalist',linestyle='-.',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        ax.set_ylim(0.4,1.0)
        ax.legend(loc='center left',bbox_to_anchor=(1,0.375),ncol=1)
        
        axTime=fig.add_axes((0.0,0.0,0.8,0.0))
        axTime.yaxis.set_visible(False)
        axTime.set_xticks(ax.get_xticks())
        def tick_function(X):
            #V = 1/(1+X)
            V=(X*3)/60
            return ["%.1f" % z for z in V]
        axTime.set_xticklabels(tick_function(ax.get_xticks()))
        axTime.set_xlim(ax.get_xlim())
        axTime.set_xlabel("Minimum session-specific recording time (minutes)")
        #https://stackoverflow.com/questions/31803817/how-to-add-second-x-axis-at-the-bottom-of-the-first-one-in-matplotlib
        
        plt.show()
        
        
        
        
        
        '''** CAN HALVE THE PRIOR FOR CONFIG PORT **'''
        
      #  fig,ax=plt.subplots();
        fig=plt.figure()
        ax=fig.add_axes((0.0,0.15,0.8,0.8))
        
      #  scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session learning')
   #     scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Dataset augmented by\nall prior user data')
   #    scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Transfer from prior user\ndata including optimisation')
   #     scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Transfer from prior user\ndata with static configuration',c='tab:red')
   #     scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Transfer from Generalist',c='tab:brown')
        scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session training\n(system config optimised\non all prior user data)',c='tab:purple')
        within_opt_2_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session training\n(system config optimised\non Session 2 user data)')
        within_opt_both_downsample_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session training\n(system config optimised on\ndownsampled prior user data)',c='tab:gray')
        topupOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session training\n(system config optimised on\njoint session-specific &\nprior user data)')
  #      plt.axhline(y=train_both_baseline_score,label='Optimised & pretrained \non all prior user data',linestyle='--',color='black')
        #   plt.axhline(y=train_2_baseline_score,label='Train session 2',linestyle='--',color='tab:green')
        #   plt.axhline(y=train_both_downsample_baseline_score,label='Train 1 + 2\n(downsampled to half)',linestyle='--',color='tab:red')
                         
    #    ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies of WITHIN-SESSION approaches over Development subjects\non reserved 33% of Session 3 data (66 gestures)',loc='left')
        ax.set_xlabel('# Session 3 gestures used for learning')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        

       # plt.axhline(y=0.723,label='Proxy* Generalist',linestyle='-.',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        ax.set_ylim(0.4,1.0)
        ax.legend(loc='center left',bbox_to_anchor=(1,0.375),ncol=1)
        
        axTime=fig.add_axes((0.0,0.0,0.8,0.0))
        axTime.yaxis.set_visible(False)
        axTime.set_xticks(ax.get_xticks())
        def tick_function(X):
            #V = 1/(1+X)
            V=(X*3)/60
            return ["%.1f" % z for z in V]
        axTime.set_xticklabels(tick_function(ax.get_xticks()))
        axTime.set_xlim(ax.get_xlim())
        axTime.set_xlabel("Minimum session-specific recording time (minutes)")
        #https://stackoverflow.com/questions/31803817/how-to-add-second-x-axis-at-the-bottom-of-the-first-one-in-matplotlib
        
        plt.show()
        
        
        
        
        
        
        '''** INTERESTING **'''
        
        fig,ax=plt.subplots();
        
        
        if 0:
            scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only',yerr='std',capsize=5)
            scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented',yerr='std',capsize=5)
            scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\ntrained on 1&2',yerr='std',capsize=5)
            scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nwithout opt',yerr='std',capsize=5)
            scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only\nwithout opt',yerr='std',capsize=5)
            scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist',yerr='std',capsize=5)
        else:
            scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 within-session')
            scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmenting\nsessions 1 + 2')
           # scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom 1 + 2 incl\nopt for transfer')
            scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom static 1 + 2',c='tab:red')
            scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist',c='tab:brown')
            scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 trained\noptimised for 1+2',c='tab:purple')
         #   within_opt_2_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 train\nOptimised for 2')
            within_opt_both_downsample_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 trained\noptimised for 1+2\n(downsampled to half)',c='tab:gray')
          #  topupOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 train\nOptimised for 3 topped\nup by 1+2 to 200 total')
            plt.axhline(y=train_both_baseline_score,label='Optimised & Trained \non sessions 1 + 2',linestyle='--',color='black')
         #   plt.axhline(y=train_2_baseline_score,label='Train session 2',linestyle='--',color='tab:green')
         #   plt.axhline(y=train_both_downsample_baseline_score,label='Train 1 + 2\n(downsampled to half)',linestyle='--',color='tab:red')
                          
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('INTERESTING Mean accuracies over Development subjects\non reserved 33% of session 3 (66 gestures)',loc='left')
        ax.set_xlabel('# Session 3 gestures (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        #below is 869066 taken from RQ2's A1_FullBespoke (in "provis")
        # but MIGHT actually be 865025 if we used D1b_RolloffStable5trials (which we use for RQ2 Hyp1)
        # OR 866105 if we use D1a_AugStable_with_errorCalcs
      #  plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        # NOW we take it from RQ2 rolloffStable5trials, but corrected (previously 5 trials werent different)
  #      plt.axhline(y=noAugRQ2['mean'][0],label='RQ2 Bespoke\n(Not session-split)',linestyle='-.',color='pink')
  #      plt.axhline(y=noAugRQ2['mean'][0]+noAugRQ2['std'][0],linestyle=':',color='pink')
  #      plt.axhline(y=noAugRQ2['mean'][0]-noAugRQ2['std'][0],linestyle=':',color='pink')
        plt.axhline(y=0.723,label='Proxy* Generalist',linestyle='-.',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        #ax.set_ylim(0.4,0.9)
        #plt.axhline(y=0.7475,label='Train 1&2\n(no cal) avg',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),ncol=1)
        plt.show()
        
        
        
        '''** VIABLE **'''
        
        fig,ax=plt.subplots();
        
        
        if 0:
            scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only',yerr='std',capsize=5)
            scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented',yerr='std',capsize=5)
            scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\ntrained on 1&2',yerr='std',capsize=5)
            scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nwithout opt',yerr='std',capsize=5)
            scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only\nwithout opt',yerr='std',capsize=5)
            scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist',yerr='std',capsize=5)
        else:
            scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 within-session')
          #  scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmenting\nsessions 1 + 2')
           # scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom 1 + 2 incl\nopt for transfer')
            scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom static 1 + 2',c='tab:red')
            scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist',c='tab:brown')
         #   scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 trained\noptimised for 1+2',c='tab:purple')
         #   within_opt_2_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 train\nOptimised for 2')
            within_opt_both_downsample_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 trained\noptimised for 1+2\n(downsampled to half)',c='tab:gray')
          #  topupOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 train\nOptimised for 3 topped\nup by 1+2 to 200 total')
            plt.axhline(y=train_both_baseline_score,label='Optimised & Trained \non sessions 1 + 2',linestyle='--',color='black')
         #   plt.axhline(y=train_2_baseline_score,label='Train session 2',linestyle='--',color='tab:green')
         #   plt.axhline(y=train_both_downsample_baseline_score,label='Train 1 + 2\n(downsampled to half)',linestyle='--',color='tab:red')
                          
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('VIABLE Mean accuracies over Development subjects\non reserved 33% of session 3 (66 gestures)',loc='left')
        ax.set_xlabel('# Session 3 gestures (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        #below is 869066 taken from RQ2's A1_FullBespoke (in "provis")
        # but MIGHT actually be 865025 if we used D1b_RolloffStable5trials (which we use for RQ2 Hyp1)
        # OR 866105 if we use D1a_AugStable_with_errorCalcs
      #  plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        # NOW we take it from RQ2 rolloffStable5trials, but corrected (previously 5 trials werent different)
  #      plt.axhline(y=noAugRQ2['mean'][0],label='RQ2 Bespoke\n(Not session-split)',linestyle='-.',color='pink')
  #      plt.axhline(y=noAugRQ2['mean'][0]+noAugRQ2['std'][0],linestyle=':',color='pink')
  #      plt.axhline(y=noAugRQ2['mean'][0]-noAugRQ2['std'][0],linestyle=':',color='pink')
        plt.axhline(y=0.723,label='Proxy* Generalist',linestyle='-.',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        #ax.set_ylim(0.4,0.9)
        #plt.axhline(y=0.7475,label='Train 1&2\n(no cal) avg',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),ncol=1)
        plt.show()
        
        
        
        
        
        '''** ALL CANDIDATES **'''
        
      #  fig,ax=plt.subplots();
        fig=plt.figure()
        ax=fig.add_axes((0.0,0.15,0.8,0.8))
        
        scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session\nlearning')
       # scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Dataset augmented by\nall prior user data')
       # scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Transfer from prior user\ndata including optimisation')
        scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Transfer from\nprior user data\nwith static\nconfiguration',c='tab:red')
        scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Transfer from\nGeneralist',c='tab:brown')
      #  scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session training\n(system config optimised\non all prior user data)',c='tab:purple')
      #  within_opt_2_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session training\n(system config optimised\non Session 2 user data)')
        within_opt_both_downsample_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session\ntraining (system\nconfig optimised\non downsampled\nprior user data)',c='tab:gray')
      #  topupOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session training\n(system config optimised on\njoint session-specific &\nprior user data)')
        plt.axhline(y=train_both_baseline_score,label='Pretrained on\nprior user data',linestyle='--',color='black')
        #   plt.axhline(y=train_2_baseline_score,label='Train session 2',linestyle='--',color='tab:green')
        #   plt.axhline(y=train_both_downsample_baseline_score,label='Train 1 + 2\n(downsampled to half)',linestyle='--',color='tab:red')
      
    #    ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies of candidate approaches over Development subjects\non reserved 33% of Session 3 data (66 gestures)',loc='left')
        ax.set_xlabel('# Session 3 gestures used for learning')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        

        plt.axhline(y=0.723,label='Proxy* Generalist',linestyle='-.',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        ax.set_ylim(0.4,1.0)
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),ncol=1)
        
        axTime=fig.add_axes((0.0,0.0,0.8,0.0))
        axTime.yaxis.set_visible(False)
        axTime.set_xticks(ax.get_xticks())
        def tick_function(X):
            #V = 1/(1+X)
            V=(X*3)/60
            return ["%.1f" % z for z in V]
        axTime.set_xticklabels(tick_function(ax.get_xticks()))
        axTime.set_xlim(ax.get_xlim())
        axTime.set_xlabel("Minimum session-specific recording time (minutes)")
        #https://stackoverflow.com/questions/31803817/how-to-add-second-x-axis-at-the-bottom-of-the-first-one-in-matplotlib
        
        plt.show()
        
        
        
        
        
        
        
        
        '''
        fig,ax=plt.subplots();
        scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 within-session')
        scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmenting 1+2')
        scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\n(optimised for transfer)')
        scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\n(from naive 1+2 model)')
        
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies over subjects on reserved 33% of session 3 (66 gests)')
        ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
       # plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        #plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        plt.axhline(y=0.7475,label='Trained & opt for\n1+2 (no cal)',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
        plt.show()
        '''
        
        
        '''
        fig,ax=plt.subplots();
        scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 within-session')
        scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmenting 1+2')
        scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\n(optimised for transfer)')
        scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\n(from naive 1+2 model)')
        
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies over subjects on reserved 33% of session 3 (66 gests)')
        ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
       # plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        #plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        ax.set_ylim(0.6,0.9)
        #ax.set_xlim(0,135)
        plt.axhline(y=0.7475,label='Trained on & opt-ed for\n1+2 (no cal)',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
        plt.show()
        '''
        
        
        
        
        fig,ax=plt.subplots();
        scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 within-session')
        scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmenting 1+2')
        scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\n(optimised for transfer)')
        scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\n(from naive 1+2 model)')
        scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 within-session\nwithout opt')
        scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist')
        
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies over subjects on reserved 33% of session 3 (66 gests)')
        ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
       # plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0],label='RQ2 Bespoke\n(Not session-split)',linestyle='--',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0]+noAugRQ2['std'][0],linestyle=':',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0]-noAugRQ2['std'][0],linestyle=':',color='pink')
        #plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        ax.set_ylim(0.6,0.9)
        #ax.set_xlim(0,135)
        plt.axhline(y=0.7475,label='Trained on & opt-ed for\n1+2 (no cal)',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
        
        
        tPerGest=4
        nCalibTotal=134
        tTotal=nCalibTotal*tPerGest
        tTotal_mins = tTotal/60
        t_save_mins = 6
        t_save=t_save_mins*60
        nCalibSave=np.floor((t_save/tPerGest)/4)*4
        #https://stackoverflow.com/questions/14892619/annotating-dimensions-in-matplotlib
        #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axvspan.html
        #https://stackoverflow.com/questions/36423221/matplotlib-axvspan-solid-fill
        
        #plt.axhspan(ymin=np.max(scores_sessiononly_agg)['mean']-0.03,ymax=np.max(scores_sessiononly_agg)['mean'],
        accept_loss = 1
        maxacc=0.8532196970500001
        min_acceptable=0.8532196970500001-(accept_loss/100)
        plt.axhspan(ymin=min_acceptable,ymax=0.8532196970500001, #ymin=0.8232196970500001
                    xmin=0.175,xmax=132/138,linestyle=':',lw=0.5,alpha=0.5,color='gray')
        ax.annotate("",xy=(138*0.1375,min_acceptable*0.995),#could do just 0.125 with xycoords=('axes fraction','data'))
                    xytext=(138*0.1375,0.8532196970500001/0.999),
                    arrowprops=dict(arrowstyle='->'))
    #    ax.annotate("",xy=(138*0.125,0.8232196970500001),xytext=(138*0.125,0.8532196970500001),
    #                arrowprops=dict(arrowstyle='|-|'))
       # bbox=dict(fc="white", ec="none")
        ax.text(138*0.05,(min_acceptable+0.8532196970500001)/2,
                f"Acceptable\naccuracy\nloss: {accept_loss}%", ha="center", va="center",size='small')#, bbox=bbox)
        
        
        plt.axvspan(132-nCalibSave,132,ymin=0.15,ymax=np.max(scores_sessiononly_agg)['mean']*0.99,
                    linestyle=':',lw=0.5,alpha=0.5,color='gray')
        
        ax.annotate("",xy=(132/0.995,0.15),xycoords=('data','axes fraction'),
                    xytext=((132-nCalibSave)*0.985,0.15),textcoords=('data','axes fraction'),
                    arrowprops=dict(arrowstyle='<-'))
       # bbox=dict(fc="white", ec="none")
        ax.text(132-(nCalibSave*0.5),0.625,
                f"Desired time reduction: {t_save_mins} mins", ha="center", va="center",size='small')#, bbox=bbox)
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
    
    

    