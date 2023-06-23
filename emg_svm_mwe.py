# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:51:35 2023

@author: pritcham
"""

import numpy as np
import pandas as pd
import statistics as stats
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, log_loss, confusion_matrix, ConfusionMatrixDisplay #plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import random
import matplotlib.pyplot as plt
import time
import pickle as pickle

idx_to_gestures = {1.:'close',2.:'open',3.:'grasp',4.:'lateral',5.:'tripod',0.:'neutral',6.:'cylindrical',7.:'spherical',8.:'lumbrical',9.:'rest'}

def drop_ID_cols(csv_dframe):
    IDs=csv_dframe.filter(regex='^ID_').columns
    csv_dframe=csv_dframe.drop(IDs,axis='columns')
    return csv_dframe

def prob_dist(model,values):
    distro = model.predict_proba(values)
    distro[distro==0]=0.00001
    return distro

def pred_from_distro(labels,distro):
    pred=int(np.argmax(distro))
    label=labels[pred]
    return label

def confmat(y_true,y_pred,labels,modelname="",testset="",title=""):
    '''y_true = actual classes, y_pred = predicted classes,
    labels = names of class labels'''
    conf=confusion_matrix(y_true,y_pred,labels=labels,normalize='true')
    cm=ConfusionMatrixDisplay(conf,labels)
    if modelname != "" and testset != "":
        title=modelname+'\n'+testset
    fig,ax=plt.subplots()
    ax.set_title(title)
    cm.plot(ax=ax)

def classes_from_preds(targets,predlist_emg,predlist_eeg,predlist_fusion,classlabels):
    '''Convert predictions to gesture labels'''
    gest_truth=[idx_to_gestures[gest] for gest in targets]
    gest_pred_emg=[idx_to_gestures[pred] for pred in predlist_emg]
    gest_pred_eeg=[idx_to_gestures[pred] for pred in predlist_eeg]
    gest_pred_fusion=[idx_to_gestures[pred] for pred in predlist_fusion]
    gesturelabels=[idx_to_gestures[label] for label in classlabels]
    
    return gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels


def scale_feats_train(data,mode='normalise'):
    '''data is a dataframe of feats, mode = normalise or standardise'''
    if mode is None:
        return data, None
    if mode=='normalise' or mode=='normalize':
        scaler=Normalizer()
    elif mode=='standardise' or mode=='standardize':
        scaler=StandardScaler()
    cols_to_ignore=list(data.filter(regex='^ID_').keys())
    cols_to_ignore.append('Label')
    data[data.columns[~data.columns.isin(cols_to_ignore)]]=scaler.fit_transform(data[data.columns[~data.columns.isin(cols_to_ignore)]])
    return data, scaler

def scale_feats_test(data,scaler):
    '''data is a dataframe of feats, scaler is a scaler fit to training data'''
    if scaler is None:
        return data
    cols_to_ignore=list(data.filter(regex='^ID_').keys())
    cols_to_ignore.append('Label')
    data[data.columns[~data.columns.isin(cols_to_ignore)]]=scaler.fit_transform(data[data.columns[~data.columns.isin(cols_to_ignore)]])
    return data 


def sel_percent_feats_df(data,percent=15):
    target=data['Label']
    attribs=data.drop(columns=['Label'])
    selector=SelectPercentile(f_classif,percentile=percent)
    selector.fit(attribs,target)
    col_idxs=selector.get_support(indices=True)
    return col_idxs


def get_ppt_split(featset,args={'using_literature_data':True}):
    if args['using_literature_data']:
        masks=[featset['ID_pptID']== n_ppt for n_ppt in np.sort(featset['ID_pptID'].unique())] 
    return masks


def balance_set(emg_set,eeg_set):
    
    index_emg=pd.MultiIndex.from_arrays([emg_set[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=pd.MultiIndex.from_arrays([eeg_set[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg=emg_set.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg=eeg_set.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    
    emg.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    eeg['ID_stratID']=eeg['ID_pptID'].astype(str)+eeg['Label'].astype(str)
    emg['ID_stratID']=emg['ID_pptID'].astype(str)+emg['Label'].astype(str)
    
    stratsize=np.min(emg['ID_stratID'].value_counts())
    balemg = emg.groupby('ID_stratID')

    balemg=balemg.apply(lambda x: x.sample(stratsize))
    print('subsampling to ',str(stratsize),' per combo of ppt and class')
   
    #print('----------\nEMG Balanced')
    
    index_balemg=pd.MultiIndex.from_arrays([balemg[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    baleeg=eeg_set.loc[index_eeg.isin(index_balemg)].reset_index(drop=True)
   
    #print('----------\nBoth Balanced')
    
    balemg.drop(columns='ID_stratID',inplace=True)
    return balemg,baleeg

def train_SVC_Platt(train_data,args):
    kernel=args['kernel']
    C=args['svm_C']
    gamma=args['gamma']
    if kernel=='linear':
        model=SVC(C=C,kernel=kernel,probability=True)
    else:
 #       model=SVC(C=C,kernel=kernel,gamma=gamma,probability=True)
        svc=SVC(C=C,kernel=kernel,gamma=gamma)
        model=CalibratedClassifierCV(svc,cv=5)#,ensemble=False, new in 0.24 and not good anyway to compete ensemble against singular
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model

def check_base_SVC(args,train_data,test_data):
    kernel=args['kernel']
    C=args['svm_C']
    gamma=args['gamma']
    svc=SVC(C=C,kernel=kernel,gamma=gamma)
    train=train_data.values[:,:-1]
    traintargets=train_data.values[:,-1]
    svc.fit(train.astype(np.float64),traintargets)
    trainscore=svc.score(train.astype(np.float64),traintargets)
    #print(f"For gamma {gamma} and C {C}, base train score = {trainscore}")
    test=test_data.values[:,:-1]
    testtargets=test_data.values[:,-1]
    score=svc.score(test.astype(np.float64),testtargets)
    #print(f"For gamma {gamma} and C {C}, base test score = {score}")
    return trainscore,score
    

def only_EMG(emg_others,eeg_others,emg_ppt,eeg_ppt,args):
    '''TRAINING ON NON-PPT DATA (or train split of bespoke)'''
    emg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=pd.MultiIndex.from_arrays([emg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=pd.MultiIndex.from_arrays([eeg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_others=emg_others.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_others=eeg_others.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    

    '''Train EMG model'''
    emg_train=drop_ID_cols(emg_others)
    sel_cols_emg=sel_percent_feats_df(emg_train,percent=15)
    sel_cols_emg=np.append(sel_cols_emg,emg_train.columns.get_loc('Label'))
    emg_train=emg_train.iloc[:,sel_cols_emg]
    
    emg_model = train_SVC_Platt(emg_train, args['emg'])
    classlabels=emg_model.classes_   
 
 
    '''TESTING ON PPT DATA'''
    emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
 
    predlist_emg=[]
    targets=[]
     
    index_emg=pd.MultiIndex.from_arrays([emg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=pd.MultiIndex.from_arrays([eeg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg=emg_ppt.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg=eeg_ppt.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    
    if emg['Label'].equals(eeg['Label']):
        targets=emg['Label'].values.tolist()
    else:
        raise Exception('Sense check failed, target label should agree between modes')

    '''Get values from instances'''
    IDs=list(eeg.filter(regex='^ID_').keys())
    emg=emg.drop(IDs,axis='columns')
    emg=emg.iloc[:,sel_cols_emg]
    
    base_trainacc,base_testacc=check_base_SVC(args['emg'], emg_train, emg)
    
    emgvals=emg.drop(['Label'],axis='columns').values    
    
    '''Get EMG Predictions'''
    distros_emg=prob_dist(emg_model,emgvals)
    for distro in distros_emg:
        pred_emg=pred_from_distro(classlabels,distro)
        predlist_emg.append(pred_emg)
    
    if args['get_train_acc']:
        predlist_emgtrain=[]
        traintargs=emg_train['Label'].values.tolist()
        emgtrainvals=emg_train.drop('Label',axis='columns') #why DOESNT this need to be .values?
        distros_emgtrain=prob_dist(emg_model,emgtrainvals)
        for distro in distros_emgtrain:
            pred_emgtrain=pred_from_distro(classlabels,distro)
            predlist_emgtrain.append(pred_emgtrain)
        return targets, predlist_emg, predlist_emg, predlist_emg, classlabels, traintargs, predlist_emgtrain, base_trainacc,base_testacc
   
    else:
        return targets, predlist_emg, predlist_emg, predlist_emg, classlabels
    

def function_fuse_withinppt(args):
    start=time.time()
    if not args['data_in_memory']:
        emg_set_path=args['emg_set_path']
        eeg_set_path=args['eeg_set_path']
    
        emg_set=pd.read_csv(emg_set_path,delimiter=',')
        eeg_set=pd.read_csv(eeg_set_path,delimiter=',')
    else:
        emg_set=args['emg_set']
        eeg_set=args['eeg_set']
    if not args['prebalanced']: 
        emg_set,eeg_set=balance_set(emg_set,eeg_set)
    
    eeg_masks=get_ppt_split(eeg_set,args)
    emg_masks=get_ppt_split(emg_set,args)
    
    accs=[]
    emg_accs=[]
    eeg_accs=[]
    
    train_accs=[]
    
    base_trainaccs=[]
    base_testaccs=[]
    
    for idx,emg_mask in enumerate(emg_masks):
        eeg_mask=eeg_masks[idx]
        
        emg_ppt = emg_set[emg_mask]
        eeg_ppt = eeg_set[eeg_mask]
        
        emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        
        index_emg=pd.MultiIndex.from_arrays([emg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
        index_eeg=pd.MultiIndex.from_arrays([eeg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
        emg_ppt=emg_ppt.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
        eeg_ppt=eeg_ppt.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
        
        
        eeg_ppt['ID_stratID']=eeg_ppt['ID_run'].astype(str)+eeg_ppt['Label'].astype(str)+eeg_ppt['ID_gestrep'].astype(str)
        emg_ppt['ID_stratID']=emg_ppt['ID_run'].astype(str)+eeg_ppt['Label'].astype(str)+eeg_ppt['ID_gestrep'].astype(str)
        random_split=random.randint(0,100)
        
        if not emg_ppt['ID_stratID'].equals(eeg_ppt['ID_stratID']):
            raise ValueError('EMG & EEG performances misaligned')
        gest_perfs=emg_ppt['ID_stratID'].unique()
        gest_strat=pd.DataFrame([gest_perfs,[perf.split('.')[1][-1] for perf in gest_perfs]]).transpose()
        train_split,test_split=train_test_split(gest_strat,test_size=0.33,random_state=random_split,stratify=gest_strat[1])
        
        eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(train_split[0])]
        eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split[0])]
        emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(train_split[0])]
        emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]

      
        if args['fusion_alg']=='just_emg':
            
            if args['scalingtype']:
                emg_train,emgscaler=scale_feats_train(emg_train,args['scalingtype'])
                eeg_train,eegscaler=scale_feats_train(eeg_train,args['scalingtype'])
                emg_test=scale_feats_test(emg_test,emgscaler)
                eeg_test=scale_feats_test(eeg_test,eegscaler)
            
            if not args['get_train_acc']:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=only_EMG(emg_train, eeg_train, emg_test, eeg_test, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train, base_trainacc,base_testacc=only_EMG(emg_train, eeg_train, emg_test, eeg_test, args)
        
               
        gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels=classes_from_preds(targets,predlist_emg,predlist_eeg,predlist_fusion,classlabels)
        
        if args['plot_confmats']:
            gesturelabels=[idx_to_gestures[label] for label in classlabels]
            confmat(gest_truth,gest_pred_emg,gesturelabels,title='EMG')
            
        emg_accs.append(accuracy_score(gest_truth,gest_pred_emg))
        eeg_accs.append(accuracy_score(gest_truth,gest_pred_eeg))
        accs.append(accuracy_score(gest_truth,gest_pred_fusion))
        
        base_trainaccs.append(base_trainacc)
        base_testaccs.append(base_testacc)
        
        if args['get_train_acc']:
            train_truth=[idx_to_gestures[gest] for gest in traintargs]
            train_preds=[idx_to_gestures[pred] for pred in predlist_train]
            if args['plot_confmats']:
                confmat(train_truth,train_preds,gesturelabels,title='Train EMG')
            train_accs.append(accuracy_score(train_truth,train_preds))
        else:
            train_accs.append(0)
        
    mean_acc=stats.mean(accs)
    mean_emg=stats.mean(emg_accs)
    median_emg=stats.median(emg_accs)
    mean_train_acc=stats.mean(train_accs)
    mean_base_acc=stats.mean(base_testaccs)
    mean_base_train=stats.mean(base_trainaccs)
    end=time.time()
    return {
        'loss': 1-mean_acc,
        'emg_mean_acc_prob':mean_emg,
        'emg_median_acc':median_emg,
        'emg_accs':emg_accs,
        'eeg_accs':eeg_accs,
        'fusion_accs':accs,
        'mean_train_acc_prob':mean_train_acc,
        'emg_mean_base':mean_base_acc,
        'mean_train_base':mean_base_train,
        'elapsed_time':end-start,}

def plot_multi_runbest_df(trials,stats,runbest,ylower=0,yupper=1,showplot=True):
    fig,ax=plt.subplots()
    for stat in stats:
        trials[stat].plot(ax=ax,label=stat)
     #   ax.plot(range(1, len(trials) + 1),
      #          [trials[stat].T],
       #         label=(stat))
        #https://www.kaggle.com/code/fanvacoolt/tutorial-on-hyperopt?scriptVersionId=12981074&cellId=97
    #ax.set(title=stat+' over optimisation iterations')
    if runbest is not None:
        best=np.fmax.accumulate(trials[runbest])
        best.plot(ax=ax,label='running best')
    ax.legend()#loc='upper center')
    ax.set_ylim(ylower,yupper)
    if showplot:
        plt.show()
    return fig

def boxplot_param(df_in,param,target,ylower=0,yupper=1,showplot=True):
    fig,ax=plt.subplots()
    dataframe=df_in.copy()
    if isinstance(dataframe[param][0],list):
        dataframe[param]=dataframe[param].apply(lambda x: x[0])
    dataframe.boxplot(column=target,by=param,ax=ax,showmeans=True)
    ax.set_ylim(ylower,yupper)
    if showplot:
        plt.show()
    return fig

if __name__ == '__main__':

    trialmode='WithinPpt'
    
    results=[]
    
    args={'emg':{'emg_model_type':'SVM_PlattScale',
                 'kernel':'rbf',
                 'svm_C':None, 
                 'gamma':None,
                 },
      'fusion_alg':'just_emg',
      'emg_set_path':'H:/Jeong11tasks_data/jeong_EMGfeats.csv',
      'eeg_set_path':'H:/Jeong11tasks_data/DUMMY_EEG.csv',
      'using_literature_data':True,
      'data_in_memory':False,
      'prebalanced':False,
      'scalingtype':'standardise',
      'plot_confmats':False,
      'get_train_acc':True,
      'trialmode':trialmode,
      }
    
    emg_set=pd.read_csv(args['emg_set_path'],delimiter=',')
    eeg_set=pd.read_csv(args['eeg_set_path'],delimiter=',')
    emg_set,eeg_set=balance_set(emg_set,eeg_set)
    args.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True})
    
    args['emg'].update({'svm_C':88.202,
                       'gamma':0.2435})
    res=function_fuse_withinppt(args)
    res.update({'svm_C':88.202,
                       'gamma':0.2435})
    results.append(res)
    
    args['emg'].update({'svm_C':0.534,
                       'gamma':0.161})
    res=function_fuse_withinppt(args)
    res.update({'svm_C':0.534,
                       'gamma':0.161})
    results.append(res)
    
    args['emg'].update({'svm_C':0.2979,
                       'gamma':65.255})
    res=function_fuse_withinppt(args)
    res.update({'svm_C':0.2979,
                       'gamma':65.255})
    results.append(res)
    
    args['emg'].update({'svm_C':91.88,
                       'gamma':1.289})
    res=function_fuse_withinppt(args)
    res.update({'svm_C':91.88,
                       'gamma':1.289})
    results.append(res)
    
    args['emg'].update({'svm_C':9.101,
                       'gamma':5.689})
    res=function_fuse_withinppt(args)
    res.update({'svm_C':9.101,
                       'gamma':5.689})
    results.append(res)
    result_table=pd.DataFrame(results)
    
    plot_multi_runbest_df(result_table,['emg_mean_acc_prob','mean_train_acc_prob','emg_mean_base','mean_train_base'],'emg_mean_acc_prob',ylower=0,yupper=1,showplot=True)