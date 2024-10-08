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
    if model.__class__.__name__ == 'RandomForestClassifier':
        model.set_params(n_estimators=model.n_estimators+10)
    model.fit(train.astype(np.float64),targets)
    return model

def partfit_model(model,calib_data):
    train=calib_data.values[:,:-1]
    targets=calib_data.values[:,-1]
    model.partial_fit(train.astype(np.float64),targets)
    return model

def warm_cal_models(emg_model,eeg_model,emg_calib,eeg_calib,args):
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
    if fuser.__class__.__name__ == 'RandomForestClassifier':
        fuser.set_params(n_estimators=fuser.n_estimators+10)
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
    emg_model,eeg_model,fuser,onehotEncoder=train_fuse_stack(emg_calib, eeg_calib, args,emg_model,eeg_model,fuser,heat='warm')
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
    load_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\D1_1_Warmstart_NoAppend_final_resMinimal - Copy.csv"
    
    load_sessiononly_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\C1_Session3onlyfinal_resMinimal - Copy.csv"
    load_aug_unadjusted=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\B1_AugPipelinefinal_resMinimal - Copy.csv"
    
    load_aug_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\B1_1_AugAdjustedSplit_final_resMinimal - Copy.csv"
    
    
    load_RF_bad_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\D1_1_Warmstart_NoAppend_final_resMinimal - Copy.csv"
    
    load_RF_fixed_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\D1a_Warmstart_FixRF_final_resMinimal - Copy.csv"
    load_res_path=load_RF_fixed_path
    
    systemUnderTest = 'D1a_Warmstart_FixRF'
    
    testset_size = 0.33
        
    if systemUnderTest == 'D1_1_Warmstart_NoAppend':
        
        calib_levels = [8/134,20/134,40/134,60/134,72/134,100/134,120/134,132/134]
        # can't do 4/134 as not left with enough calib data to warmstart and use for opt target
        calib_levels = [80/134]
                
        feats_method='calib'
        opt_method='calib'
        train_method='calib'
        
        train_session='both'
        
        calib_levels = np.array([round(scale/(4/134))*(4/134) for scale in calib_levels])
        
    elif systemUnderTest == 'D1a_TEST_Warmstart_FixRF':
        calib_levels = [8/134,20/134,60/134,120/134]
        # can't do 4/134 as not left with enough calib data to warmstart and use for opt target
        #skip 72, 80, 40, 100, 132
                
        feats_method='calib'
        opt_method='calib'
        train_method='calib'
        
        train_session='both'
        
        calib_levels = np.array([round(scale/(4/134))*(4/134) for scale in calib_levels])
        
    elif systemUnderTest == 'D1a_Warmstart_FixRF':
        calib_levels = [8/134,20/134,40/134,60/134,72/134,80/134,100/134,120/134,132/134]
        # can't do 4/134 as not left with enough calib data to warmstart and use for opt target
                
        feats_method='calib'
        opt_method='calib'
        train_method='calib'
        
        train_session='both'
        
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
        
        for calib_level in calib_levels:
                for idx,emg_mask in enumerate(emg_masks):
                    print('Calib level: ',str(calib_level),' (subject ',str(idx),' of 20)')
    
                    space=setup_warmstart_space(architecture='decision',include_svm=True)
                    
                    space.update({'l1_sparsity':0.05})
                    space.update({'l1_maxfeats':40})
                    
                    space.update({'calib_level':calib_level})
                    
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
                        trials=Trials()
                        
                        
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
                            
                        
                        gest_perfs=emg_session3['ID_stratID'].unique()
                        gest_strat=pd.DataFrame([gest_perfs,[perf.split('.')[1][-1] for perf in gest_perfs]]).transpose()
                        
                        calib_split,test_split=train_test_split(gest_strat,test_size=testset_size,
                                                              random_state=random_split,stratify=gest_strat[1])
                        
    
                        emg_test=emg_session3[emg_session3['ID_stratID'].isin(test_split[0])]
                        eeg_test=eeg_session3[eeg_session3['ID_stratID'].isin(test_split[0])]
                 
                             
                        emg_train,emgscaler=feats.scale_feats_train(emg_train,space['scalingtype'])
                        eeg_train,eegscaler=feats.scale_feats_train(eeg_train,space['scalingtype'])
                        emg_test=feats.scale_feats_test(emg_test,emgscaler)
                        eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
                                        
                        
                        if calib_level == 0:
                            emg_joint = emg_train
                            eeg_joint = eeg_train
                        else:
                            if space['calib_level'] > 130/134:
                                '''case for when we cant train_test_split, as the (unused) "test" would be < n_classes'''
                                stratsize=np.min(calib_split[1].value_counts())
                                calib_split = calib_split.groupby(1,axis=0)
                                calib_split=calib_split.apply(lambda x: x.sample(stratsize))
                            elif 0 < space['calib_level'] < 1:
                                calib_split,_=train_test_split(calib_split,train_size=space['calib_level'],random_state=random_split,stratify=calib_split[1])
                                if min(calib_split[1].value_counts()) < 2:
                                    print('calib of ' +str(space['calib_level'])+' results in < 2 performances per class')
                                    if space['calib_level']==4/134:
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
                            
    
                        if feats_method=='no calib':
                            sel_cols_emg=feats.sel_percent_feats_df(ml.drop_ID_cols(emg_train),percent=15)
                            sel_cols_emg=np.append(sel_cols_emg,ml.drop_ID_cols(emg_train).columns.get_loc('Label'))
                            sel_cols_eeg=feats.sel_feats_l1_df(ml.drop_ID_cols(eeg_train),sparsityC=space['l1_sparsity'],maxfeats=space['l1_maxfeats'])
                            sel_cols_eeg=np.append(sel_cols_eeg,ml.drop_ID_cols(eeg_train).columns.get_loc('Label')) 
                        
                        elif feats_method=='calib':                     
                            sel_cols_emg=feats.sel_percent_feats_df(ml.drop_ID_cols(emg_joint),percent=15)
                            sel_cols_emg=np.append(sel_cols_emg,ml.drop_ID_cols(emg_joint).columns.get_loc('Label'))
                            sel_cols_eeg=feats.sel_feats_l1_df(ml.drop_ID_cols(eeg_joint),sparsityC=space['l1_sparsity'],maxfeats=space['l1_maxfeats'])
                            sel_cols_eeg=np.append(sel_cols_eeg,ml.drop_ID_cols(eeg_joint).columns.get_loc('Label')) 
                        
     
                        space['sel_cols_emg']=sel_cols_emg
                        space['sel_cols_eeg']=sel_cols_eeg
                        space['subject-id']=eeg_ppt['ID_pptID'][0]                    
                        
                        if opt_method=='no calib':
                            space.update({'emg_set':emg_train,'eeg_set':eeg_train,'data_in_memory':True,'prebalanced':True})
                        elif opt_method=='calib':
                            space.update({'emg_set':emg_joint,'eeg_set':eeg_joint,'data_in_memory':True,'prebalanced':True})
                        
                        space.update({'featsel_method':feats_method})
                        space.update({'train_method':train_method})
                        space.update({'opt_method':opt_method})
                        space.update({'train_session':train_session})
                        
                        best = fmin(fuse_warmstart,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=iters,
                                trials=trials)
                        
                        winner_args=space_eval(space,best)
                        best_loss=trials.best_trial['result']['loss']
                
                        winner_args['sel_cols_emg']=sel_cols_emg
                        winner_args['sel_cols_eeg']=sel_cols_eeg 
                        winner_args['subject id']=str(int(eeg_ppt['ID_pptID'][0]))
                        
                        if n_repeats > 1:
                            winner_args['plot_confmats']=False
                        else:
                            winner_args['plot_confmats']=True
                        
                        subject_results=fusionWarm_test(emg_joint,eeg_joint,emg_test,eeg_test,winner_args)
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
               
                calib_label='_calib'+str(round(calib_level,3))
                
                picklepath=os.path.join(resultpath,(systemUnderTest+calib_label+'_resDF.pkl'))
                csvpath=os.path.join(resultpath,(systemUnderTest+calib_label+'_resMinimal.csv'))
        
                pickle.dump(results,open(picklepath,'wb'))
                scores_minimal.to_csv(csvpath)
        
        picklefullpath=os.path.join(resultpath,(systemUnderTest+'_final_resDF.pkl'))
        csvfullpath=os.path.join(resultpath,(systemUnderTest+'_final_resMinimal.csv'))

        pickle.dump(results,open(picklefullpath,'wb'))
        scores_minimal.to_csv(csvfullpath)
    else:
        scores_minimal=pd.read_csv(load_res_path,index_col=0)        
    
    if plot_results:
        scores_aug_minimal=pd.read_csv(load_aug_res_path,index_col=0)
        scores_sessiononly_minimal=pd.read_csv(load_sessiononly_res_path,index_col=0)
        
       # scores_aug_unadjusted=pd.read_csv(load_aug_unadjusted,index_col=0)
        if 0:
            scores_RF_bad=pd.read_csv(load_RF_bad_path,index_col=0)
            #scores_RF_bad=scores_RF_bad[scores_RF_bad['subject id'].isin([2,3,4,5,7])]
            scores_RF_fixed=pd.read_csv(load_RF_fixed_path,index_col=0)
            #scores_RF_fixed=scores_RF_fixed[scores_RF_fixed['subject id'].isin([2,3,4,5,7])]
            
            nGest=4
            nRepsPerGest=50
            nInstancePerGest=4
            trainsplitSize=2/3
            scores_RF_bad['calib_level_wholegests']=scores_RF_bad['calib_level']*(1-testset_size)*nGest*nRepsPerGest
            scores_RF_fixed['calib_level_wholegests']=scores_RF_fixed['calib_level']*(1-testset_size)*nGest*nRepsPerGest
            
            fig,ax=plt.subplots();
            scores_RF_bad_agg=scores_RF_bad.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
            scores_RF_bad_agg=scores_RF_bad_agg.round({'calib_level_wholegests':5})
            
            scores_RF_fixed_agg=scores_RF_fixed.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
            scores_RF_fixed_agg=scores_RF_fixed_agg.round({'calib_level_wholegests':5})
            
            scores_RF_bad_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Without proper RF xfer',yerr='std',capsize=5)
            scores_RF_fixed_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='With 10 extra RF trees',yerr='std',capsize=5)
            
            
     #       scores_aug_unadjusted['calib_level_wholegests']=scores_aug_unadjusted['calib_level']*(1-testset_size)*nGest*nRepsPerGest
     #       aug_unadjust_agg=scores_aug_adjusted.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
     #       aug_unadjust_agg=aug_adjust_agg.round({'calib_level_wholegests':5})
     #       aug_unadjust_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented\nadjusted split')
            
    
            
           # ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
            plt.title('Mean accuracies over subjects on reserved 33% of session 3 (66 gests)')
            ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
            ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
            
            plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
            plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
            #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
            ax.set_ylim(0.5,0.95)
            plt.axhline(y=0.7475,label='Train 1&2\n(no cal) avg',linestyle='--',color='black')
            ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
            plt.show()
            
            raise
            
        
        scores_minimal=scores_minimal.round({'augment_scale':5})
        
        '''
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
                ax.set_xlabel('Proportion of non-subject data for initial training')
                ax.legend(title='Proportion of subject data')
                plt.show()
            if 1:
                fig,ax=plt.subplots();
                plt.rcParams['figure.dpi'] = 150 # DEFAULT IS 100
                subjScore.pivot(index='rolloff_factor',columns='augment_scale',values='mean').plot(kind='bar',ax=ax,rot=0,capsize=5,
                                                                                                    yerr=subjScore.pivot(index='rolloff_factor',columns='augment_scale',values='std'))
                ax.set_ylim(np.floor(subj['fusion_acc'].min()/0.05)*0.05,np.ceil(subj['fusion_acc'].max()/0.05)*0.05)
                plt.title('Subject '+str(ppt))
                ax.set_xlabel('Proportion of subject data calibrating')
                
                plt.axhline(y=gen_dev_accs[int(ppt)],label='Generalist',linestyle='--',color='gray')
                ax.legend(title='Proportion of non-subject data\n for initial training')
                plt.show()
           ''' 
        
        
        '''
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
            ax.set_xlabel('Proportion of non-subj augmenting Everything')
            plt.show()
            
        print('*****\nHeavily affected by randomness, BUT I think it may be in part the',
              'randomness of which bits of non-subj are chosen to be added. IE not randomness in model etc causing',
              'effect where there is none, but rather randomness as to how effective it will be dependent on',
              'the non-subj data that is most helpful (or most helpful to this subj) which one could',
              'theoretically identify statically or find way to auto identify.\n*****')
    
    #chance we confuse it eg that by adding non subject within opt, all its doing is learning to firstly
        #ignore the bits of training data that are non-subject, then learn helpful things from the subject data
    
        '''

        
        
        nGest=4
        nRepsPerGest=50
        nInstancePerGest=4
        trainsplitSize=2/3
        scores_minimal['calib_level_instances']=scores_minimal['calib_level']*(1-testset_size)*nGest*nRepsPerGest*nInstancePerGest
        scores_minimal['calib_level_wholegests']=scores_minimal['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        scores_minimal['calib_level_pergest']=scores_minimal['calib_level']*(1-testset_size)*nRepsPerGest
        
                
        scores_aug_minimal['calib_level_wholegests']=scores_aug_minimal['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        scores_sessiononly_minimal['calib_level_wholegests']=scores_sessiononly_minimal['calib_level']*(1-testset_size)*nGest*nRepsPerGest

        
        
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
        
        plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        plt.axhline(y=0.7475,label='Train both\n(no cal) avg',linestyle='--',color='black')
        ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        plt.show()
        
        
        
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
        
        
        
        
        fig,ax=plt.subplots();
        scores_agg=scores_minimal.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_agg=scores_agg.round({'calib_level_wholegests':5})
        
        scores_aug_agg=scores_aug_minimal.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_aug_agg=scores_aug_agg.round({'calib_level_wholegests':5})
        
        scores_sessiononly_agg=scores_sessiononly_minimal.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_sessiononly_agg=scores_sessiononly_agg.round({'calib_level_wholegests':5})
        
        scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented\ntrained on 1&2')
        scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only')
        scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\ntrained on 1&2')
        
        
 #       scores_aug_unadjusted['calib_level_wholegests']=scores_aug_unadjusted['calib_level']*(1-testset_size)*nGest*nRepsPerGest
 #       aug_unadjust_agg=scores_aug_adjusted.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
 #       aug_unadjust_agg=aug_adjust_agg.round({'calib_level_wholegests':5})
 #       aug_unadjust_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented\nadjusted split')
        

        
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies over subjects on reserved 33% of session 3 (66 gests)')
        ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        #plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        plt.axhline(y=0.7475,label='Train 1&2\n(no cal) avg',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
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
    
    

    