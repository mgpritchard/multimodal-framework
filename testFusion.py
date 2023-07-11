#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 23:42:00 2022

@author: pritcham
"""

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

def process_all_data():
    
    source_data_dir='/home/michael/Documents/Aston/MultimodalFW/dataset/dev/'
    
    pptlist=['001_retake','002_retake','4','7','8','9','11','13','14']
    
    repeatlist=['001','002'] #do NOT copy any repeats needed for consistency investigation
    
    paths=[]
    for i in range(len(pptlist)):
        paths.append(comp.build_path(source_data_dir,pptlist[i].split(' ')[0]))
    
    raw_eeg_dir='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EEG/Raw'
    cropped_eeg_dir='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EEG/Cropped'
    processed_eeg_dir='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EEG/Processed'
    
    raw_emg_dir='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EMG/Raw'
    cropped_emg_dir='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EMG/Cropped'
    processed_emg_dir='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EMG/Processed'
    
    destination='/home/michael/Documents/Aston/MultimodalFW/working_dataset/'
    
    #copy all data into raw
    '''comp.build_set_separate_modes(paths, destination,
                                  flush_folders=True,
                                  emgdest=raw_emg_dir,
                                  eegdest=raw_eeg_dir)'''
    
    '''go into the Raw folder and run
    rename 's/june//' *
    #https://unix.stackexchange.com/questions/175135/how-to-rename-multiple-files-by-replacing-string-in-file-name-this-string-conta
    which will turn all 001june -> 001'''
    
    #sync and process all data
    raw_emg_files=wrangle.list_raw_files(raw_emg_dir)
    raw_eeg_files=wrangle.list_raw_files(raw_eeg_dir)
    wrangle.sync_raw_files(raw_emg_files,raw_eeg_files,
                           cropped_emg_dir,cropped_eeg_dir,
                           unicorn_time_moved=0)
    
    processed_emg_dir=tt.process_data('emg',cropped_emg_dir,overwrite=False,
                                      bf_time_moved=False,
                                      dataout=processed_emg_dir)
    
    processed_eeg_dir=tt.process_data('eeg',cropped_eeg_dir,overwrite=False,
                                      bf_time_moved=True,
                                      dataout=processed_eeg_dir)
    
    emg_featspath=os.path.join(destination,'devset_EMG/featsEMG.csv')
    eeg_featspath=os.path.join(destination,'devset_EEG/featsEEG.csv')
    
    feats_emg=feats.make_feats(processed_emg_dir,emg_featspath,'emg',period=1000)
    feats_eeg=feats.make_feats(processed_eeg_dir,eeg_featspath,'eeg',period=1000)
    
    return feats_emg,feats_eeg
    #then later can stratify at the features level

def inspect_set_balance(emg_set_path=None,eeg_set_path=None,emg_set=None,eeg_set=None):
    #emg_set_path='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EMG/featsEMG_Labelled.csv'
    #eeg_set_path='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EEG/featsEEG_Labelled.csv'
    if emg_set is None:
        if emg_set_path is None:
            raise ValueError
        emg_set=ml.pd.read_csv(emg_set_path,delimiter=',')
    emg_set.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    if emg_set_path:
        print(emg_set_path.split('/')[-1])
    else:
        print('EMG:')
    print(emg_set['Label'].value_counts())
    print(emg_set['ID_pptID'].value_counts())
    #print(emg_set['ID_run'].value_counts())
    #print(emg_set['ID_gestrep'].value_counts())
    
    if eeg_set is None:
        if eeg_set_path is None:
            raise ValueError
        eeg_set=ml.pd.read_csv(eeg_set_path,delimiter=',')
    eeg_set.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    if eeg_set_path:
        print(eeg_set_path.split('/')[-1])
    else:
        print('EEG:')
    print(eeg_set['Label'].value_counts())
    print(eeg_set['ID_pptID'].value_counts())
    #print(eeg_set['ID_run'].value_counts())
    #print(eeg_set['ID_gestrep'].value_counts())
    
    return emg_set,eeg_set

def balance_single_mode(dataset):
    dataset['ID_stratID']=dataset['ID_pptID'].astype(str)+dataset['Label'].astype(str)
    stratsize=np.min(dataset['ID_stratID'].value_counts())
    balanced = dataset.groupby('ID_stratID')
    #g.apply(lambda x: x.sample(g.size().min()))
    #https://stackoverflow.com/questions/45839316/pandas-balancing-data
    balanced=balanced.apply(lambda x: x.sample(stratsize))
    print('subsampling to ',str(stratsize),' per combo of ppt and class')
    return balanced

def balance_set(emg_set,eeg_set):
    #print('initial')
    #_,_=inspect_set_balance(emg_set=emg_set,eeg_set=eeg_set)
    emg_set.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_set.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)

    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_set[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_set[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg=emg_set.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg=eeg_set.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    
    
    eeg['ID_stratID']=eeg['ID_pptID'].astype(str)+eeg['Label'].astype(str)
    emg['ID_stratID']=emg['ID_pptID'].astype(str)+emg['Label'].astype(str)
    
    stratsize=np.min(emg['ID_stratID'].value_counts())
    balemg = emg.groupby('ID_stratID',group_keys=False)
    #g.apply(lambda x: x.sample(g.size().min()))
    #https://stackoverflow.com/questions/45839316/pandas-balancing-data
    balemg=balemg.apply(lambda x: x.sample(stratsize))
    print('subsampling to ',str(stratsize),' per combo of ppt and class')
   
    #print('----------\nEMG Balanced')
    #_,_=inspect_set_balance(emg_set=balemg,eeg_set=eeg)
    
    index_balemg=ml.pd.MultiIndex.from_arrays([balemg[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    baleeg=eeg_set.loc[index_eeg.isin(index_balemg)].reset_index(drop=True)
   
    #print('----------\nBoth Balanced')
    #_,_=inspect_set_balance(emg_set=balemg,eeg_set=baleeg)
    
    if 0:   #manual checking, almost certainly unnecessary
        for index,emgrow in balemg.iterrows():
            eegrow = baleeg[(baleeg['ID_pptID']==emgrow['ID_pptID'])
                                  & (baleeg['ID_run']==emgrow['ID_run'])
                                  & (baleeg['Label']==emgrow['Label'])
                                  & (baleeg['ID_gestrep']==emgrow['ID_gestrep'])
                                  & (baleeg['ID_tend']==emgrow['ID_tend'])]
            #syntax like the below would do it closer to a .where
            #eegrow=test_set_eeg[test_set_eeg[['ID_pptID','Label']]==emgrow[['ID_pptID','Label']]]
            if eegrow.empty:
                print('No matching EEG window for EMG window '+str(emgrow['ID_pptID'])+str(emgrow['ID_run'])+str(emgrow['Label'])+str(emgrow['ID_gestrep'])+str(emgrow['ID_tend']))
                continue
            
            TargetLabel=emgrow['Label']
            if TargetLabel != eegrow['Label'].values:
                raise Exception('Sense check failed, target label should agree between modes')
        print('checked all for window matching')
    
    balemg.drop(columns='ID_stratID',inplace=True)
    return balemg,baleeg

def identify_rejects(rejectlog=None):
    if rejectlog is None:
        Tk().withdraw()
        root="/home/michael/Documents/Aston/MultimodalFW/working_dataset/"
        rejectlog=askopenfilename(initialdir=root,title='Select log of rejected datafiles')
    rejectpaths=np.genfromtxt(rejectlog,dtype='U')
    
    rejects=[]
    for rejectfile in rejectpaths:
        directory,file=os.path.split(rejectfile)
        reject=wrangle.build_rawfile_obj(file, directory)
        rejects.append(reject)
    return rejects

def purge_rejects(rejects,featset):
    for reject in rejects:
        pptID=int(reject.ppt) if reject.ppt[-1].isdigit() else int(reject.ppt[:-1])
        rejectrows=featset.loc[(featset['ID_pptID']==pptID)
                               & (featset['Label']==reject.label)
                                 & (featset['ID_gestrep']==int(reject.count))]
        if rejectrows.empty:
            print('No matching feature windows for rejected file '+reject.filepath)
            continue
        featset.drop(rejectrows.index,inplace=True)
    return featset

def get_ppt_split(featset,args={'using_literature_data':True}):
    if args['using_literature_data']:
        masks=get_ppt_split_flexi(featset)
        return masks
    else:
        #print(featset['ID_pptID'].unique())
        msk_ppt1=(featset['ID_pptID']==1)
        msk_ppt2=(featset['ID_pptID']==15) #ppt2 retrial was labelled 15
        msk_ppt4=(featset['ID_pptID']==4)
        msk_ppt7=(featset['ID_pptID']==7)
        msk_ppt8=(featset['ID_pptID']==8)
        msk_ppt9=(featset['ID_pptID']==9)
        msk_ppt11=(featset['ID_pptID']==11)
        msk_ppt13=(featset['ID_pptID']==13)
        msk_ppt14=(featset['ID_pptID']==14)
        #return these and then as necessary to featset[mask]
        #https://stackoverflow.com/questions/33742588/pandas-split-dataframe-by-column-value
        #so can do different permutations of assembling train/test sets
        #can also invert a mask (see link above) to get the rest for all-except-n
        return [msk_ppt1,msk_ppt2,msk_ppt4,msk_ppt7,msk_ppt8,msk_ppt9,msk_ppt11,msk_ppt13,msk_ppt14]

def get_ppt_split_flexi(featset):
    masks=[featset['ID_pptID']== n_ppt for n_ppt in np.sort(featset['ID_pptID'].unique())] 
    return masks

def isolate_holdout_ppts(ppts):
    emg_set=ml.pd.read_csv(params.jeong_EMGfeats,delimiter=',')
    eeg_set=ml.pd.read_csv(params.jeong_noCSP_WidebandFeats,delimiter=',')
    emg_set,eeg_set=balance_set(emg_set,eeg_set)
    emg_masks = get_ppt_split(emg_set)
    eeg_masks = get_ppt_split(eeg_set)
    for idx, emg_mask in enumerate(emg_masks):
        if idx not in ppts:
            continue
        else:
            eeg_mask=eeg_masks[idx]
            emg=emg_set[emg_mask]
            eeg=eeg_set[eeg_mask]
            emg_set.drop(emg_set[emg_mask].index,inplace=True)
            eeg_set.drop(eeg_set[eeg_mask].index,inplace=True)
            emg.to_csv((r"H:\Jeong11tasks_data\final_dataset\holdout\emg_holdout_ppt"+str(idx+1)+'.csv'),index=False)
            eeg.to_csv((r"H:\Jeong11tasks_data\final_dataset\holdout\eeg_holdout_ppt"+str(idx+1)+'.csv'),index=False)
    inspect_set_balance(emg_set=emg_set,eeg_set=eeg_set)
    emg_set.to_csv((r"H:\Jeong11tasks_data\final_dataset\emg_set_noholdout.csv"),index=False)
    eeg_set.to_csv((r"H:\Jeong11tasks_data\final_dataset\eeg_set_noholdout.csv"),index=False)

def synchronously_classify(test_set_emg,test_set_eeg,model_emg,model_eeg,classlabels,args):
    distrolist_emg=[]
    predlist_emg=[]
    correctness_emg=[]
    
    distrolist_eeg=[]
    predlist_eeg=[]
    correctness_eeg=[]
    
    distrolist_fusion=[]
    predlist_fusion=[]
    correctness_fusion=[]
    
    targets=[]
    
    for index,emgrow in test_set_emg.iterrows():
        eegrow = test_set_eeg[(test_set_eeg['ID_pptID']==emgrow['ID_pptID'])
                              & (test_set_eeg['ID_run']==emgrow['ID_run'])
                              & (test_set_eeg['Label']==emgrow['Label'])
                              & (test_set_eeg['ID_gestrep']==emgrow['ID_gestrep'])
                              & (test_set_eeg['ID_tend']==emgrow['ID_tend'])]
        #syntax like the below would do it closer to a .where
        #eegrow=test_set_eeg[test_set_eeg[['ID_pptID','Label']]==emgrow[['ID_pptID','Label']]]
        if eegrow.empty:
            print('No matching EEG window for EMG window '+str(emgrow['ID_pptID'])+str(emgrow['ID_run'])+str(emgrow['Label'])+str(emgrow['ID_gestrep'])+str(emgrow['ID_tend']))
            continue
        
        TargetLabel=emgrow['Label']
        if TargetLabel != eegrow['Label'].values:
            raise Exception('Sense check failed, target label should agree between modes')
        
        '''Get values from instances'''
        IDs=list(emgrow.filter(regex='^ID_').keys())
        IDs.append('Label')
        emgvals=emgrow.drop(IDs).values
        eegvals=eegrow.drop(IDs,axis='columns').values
        
        '''Pass values to models'''
        
        distro_emg=ml.prob_dist(model_emg,emgvals.reshape(1,-1))
        pred_emg=ml.pred_from_distro(classlabels,distro_emg)
        distrolist_emg.append(distro_emg)
        predlist_emg.append(pred_emg)
        
        if pred_emg == TargetLabel:
            correctness_emg.append(True)
        else:
            correctness_emg.append(False)
        
        distro_eeg=ml.prob_dist(model_eeg,eegvals.reshape(1,-1))
        pred_eeg=ml.pred_from_distro(classlabels,distro_eeg)
        distrolist_eeg.append(distro_eeg)
        predlist_eeg.append(pred_eeg)
        
        if pred_eeg == TargetLabel:
            correctness_eeg.append(True)
        else:
            correctness_eeg.append(False)
        
        #distro_fusion=fusion.fuse_mean(distro_emg,distro_eeg)
        distro_fusion=fusion.fuse_select(distro_emg, distro_eeg, args)
        pred_fusion=ml.pred_from_distro(classlabels,distro_fusion)
        distrolist_fusion.append(distro_fusion)
        predlist_fusion.append(pred_fusion)
        
        if pred_fusion == TargetLabel:
            correctness_fusion.append(True)
        else:
            correctness_fusion.append(False)
            
        targets.append(TargetLabel)
    return targets, predlist_emg, correctness_emg, predlist_eeg, correctness_eeg, predlist_fusion, correctness_fusion

def synced_predict(test_set_emg,test_set_eeg,model_emg,model_eeg,classlabels,args):
  #  distrolist_emg=[]
    predlist_emg=[]
 #   distrolist_eeg=[]
    predlist_eeg=[]
   # distrolist_fusion=[]
    predlist_fusion=[]
    
    targets=[]
    
    for index,emgrow in test_set_emg.iterrows():
        eegrow = test_set_eeg[(test_set_eeg['ID_pptID']==emgrow['ID_pptID'])
                              & (test_set_eeg['ID_run']==emgrow['ID_run'])
                              & (test_set_eeg['Label']==emgrow['Label'])
                              & (test_set_eeg['ID_gestrep']==emgrow['ID_gestrep'])
                              & (test_set_eeg['ID_tend']==emgrow['ID_tend'])]
        #syntax like the below would do it closer to a .where
        #eegrow=test_set_eeg[test_set_eeg[['ID_pptID','Label']]==emgrow[['ID_pptID','Label']]]
        if eegrow.empty:
            print('No matching EEG window for EMG window '+str(emgrow['ID_pptID'])+str(emgrow['ID_run'])+str(emgrow['Label'])+str(emgrow['ID_gestrep'])+str(emgrow['ID_tend']))
            continue
        
        TargetLabel=emgrow['Label']
        if TargetLabel != eegrow['Label'].values:
            raise Exception('Sense check failed, target label should agree between modes')
        
        '''Get values from instances'''
        IDs=list(emgrow.filter(regex='^ID_').keys())
        IDs.append('Label')
        emgvals=emgrow.drop(IDs).values
        eegvals=eegrow.drop(IDs,axis='columns').values
        
        '''Pass values to models'''
        
        distro_emg=ml.prob_dist(model_emg,emgvals.reshape(1,-1))
        pred_emg=ml.pred_from_distro(classlabels,distro_emg)
       # distrolist_emg.append(distro_emg)
        predlist_emg.append(pred_emg)
        
        distro_eeg=ml.prob_dist(model_eeg,eegvals.reshape(1,-1))
        pred_eeg=ml.pred_from_distro(classlabels,distro_eeg)
       # distrolist_eeg.append(distro_eeg)
        predlist_eeg.append(pred_eeg)
        
        distro_fusion=fusion.fuse_select(distro_emg, distro_eeg, args)
        pred_fusion=ml.pred_from_distro(classlabels,distro_fusion)
       # distrolist_fusion.append(distro_fusion)
        predlist_fusion.append(pred_fusion)
            
        targets.append(TargetLabel)
    return targets, predlist_emg, predlist_eeg, predlist_fusion

def refactor_synced_predict(test_set_emg,test_set_eeg,model_emg,model_eeg,classlabels,args, chosencolseeg=None, chosencolsemg=None, get_distros=False):

    predlist_emg=[]
    predlist_eeg=[]
    predlist_fusion=[]
    targets=[]
    
    index_emg=ml.pd.MultiIndex.from_arrays([test_set_emg[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([test_set_eeg[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg=test_set_emg.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg=test_set_eeg.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    
    if emg['Label'].equals(eeg['Label']):
        targets=emg['Label'].values.tolist()
    else:
        raise Exception('Sense check failed, target label should agree between modes')
        
    '''Get values from instances'''
    IDs=list(emg.filter(regex='^ID_').keys())
    emg=emg.drop(IDs,axis='columns')
    eeg=eeg.drop(IDs,axis='columns')
    
    if chosencolseeg is not None:
        eeg=eeg.iloc[:,chosencolseeg]
    if chosencolsemg is not None:
        emg=emg.iloc[:,chosencolsemg]
    emgvals=emg.drop(['Label'],axis='columns').values
    eegvals=eeg.drop(['Label'],axis='columns').values
    #IDs.append('Label')
    #emgvals=emg.drop(IDs,axis='columns').values
    #eegvals=eeg.drop(IDs,axis='columns').values
    
    '''Pass values to models'''
    
    distros_emg=ml.prob_dist(model_emg,emgvals)
    predlist_emg=ml.predlist_from_distrosarr(classlabels,distros_emg)

    distros_eeg=ml.prob_dist(model_eeg,eegvals)
    predlist_eeg=ml.predlist_from_distrosarr(classlabels,distros_eeg)

    distros_fusion=fusion.fuse_select(distros_emg, distros_eeg, args)
    predlist_fusion=ml.predlist_from_distrosarr(classlabels,distros_fusion)
    
    if get_distros:
        return targets, predlist_emg, predlist_eeg, predlist_fusion, distros_emg, distros_eeg, distros_fusion
    else:
        return targets, predlist_emg, predlist_eeg, predlist_fusion, None, None, None

def evaluate_results(targets, predlist_emg, correctness_emg, predlist_eeg, correctness_eeg, predlist_fusion, correctness_fusion, classlabels, plot_confmats=False):
    '''Evaluate accuracy'''
    accuracy_emg = sum(correctness_emg)/len(correctness_emg)
    print('EMG accuracy: '+ str(accuracy_emg))
    
    accuracy_eeg = sum(correctness_eeg)/len(correctness_eeg)
    print('EEG accuracy: '+str(accuracy_eeg))
    
    accuracy_fusion = sum(correctness_fusion)/len(correctness_fusion)
    print('Fusion accuracy: '+str(accuracy_fusion))
    
    '''Convert predictions to gesture labels'''
    gest_truth=[params.idx_to_gestures[gest] for gest in targets]
    gest_pred_emg=[params.idx_to_gestures[pred] for pred in predlist_emg]
    gest_pred_eeg=[params.idx_to_gestures[pred] for pred in predlist_eeg]
    gest_pred_fusion=[params.idx_to_gestures[pred] for pred in predlist_fusion]
    gesturelabels=[params.idx_to_gestures[label] for label in classlabels]
    
    if plot_confmats:
        '''Produce confusion matrix'''
        tt.confmat(gest_truth,gest_pred_emg,gesturelabels)
        tt.confmat(gest_truth,gest_pred_eeg,gesturelabels)
        tt.confmat(gest_truth,gest_pred_fusion,gesturelabels)
    
    return accuracy_emg,accuracy_eeg,accuracy_fusion

def classes_from_preds(targets,predlist_emg,predlist_eeg,predlist_fusion,classlabels):
    '''Convert predictions to gesture labels'''
    gest_truth=[params.idx_to_gestures[gest] for gest in targets]
    gest_pred_emg=[params.idx_to_gestures[pred] for pred in predlist_emg]
    gest_pred_eeg=[params.idx_to_gestures[pred] for pred in predlist_eeg]
    gest_pred_fusion=[params.idx_to_gestures[pred] for pred in predlist_fusion]
    gesturelabels=[params.idx_to_gestures[label] for label in classlabels]
    
    return gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels

def plot_confmats(gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels):
        '''Produce confusion matrix'''
        tt.confmat(gest_truth,gest_pred_emg,gesturelabels)
        tt.confmat(gest_truth,gest_pred_eeg,gesturelabels)
        tt.confmat(gest_truth,gest_pred_fusion,gesturelabels)
    #CAN you have a consistent gradation of the colour heatmap across confmats?
    #ie yellow is always a fixed % not relative to the highest in that given
    #confmat

def train_models_opt(emg_train_set,eeg_train_set,args):
    emg_model_type=args['emg']['emg_model_type']
    eeg_model_type=args['eeg']['eeg_model_type']
    emg_model = ml.train_optimise(emg_train_set, emg_model_type, args['emg'])
    eeg_model = ml.train_optimise(eeg_train_set, eeg_model_type, args['eeg'])
    return emg_model,eeg_model

def function_fuse_pptn(args,n,plot_confmats=False,emg_set_path=None,eeg_set_path=None):
    
    if emg_set_path is None:
        if args['using_literature_data']:
            emg_set_path=params.emg_waygal
        else:
            emg_set_path=params.emg_set_path_for_system_tests
    if eeg_set_path is None:
         if args['using_literature_data']:
            eeg_set_path=params.eeg_waygal
         else:
            eeg_set_path=params.eeg_set_path_for_system_tests
    
    emg_set=ml.pd.read_csv(emg_set_path,delimiter=',')
    eeg_set=ml.pd.read_csv(eeg_set_path,delimiter=',')
    
    emg_set,eeg_set=balance_set(emg_set,eeg_set)
    
    eeg_masks=get_ppt_split(eeg_set,args)
    emg_masks=get_ppt_split(emg_set,args)
    
    emg_mask_n=emg_masks[n-1]
    eeg_mask_n=eeg_masks[n-1]
    
    emg_ppt = emg_set[emg_mask_n]
    emg_others = emg_set[~emg_mask_n]
    eeg_ppt = eeg_set[eeg_mask_n]
    eeg_others = eeg_set[~eeg_mask_n]
    #emg_ppt=ml.drop_ID_cols(emg_ppt)
    emg_others=ml.drop_ID_cols(emg_others)
    #eeg_ppt=ml.drop_ID_cols(eeg_ppt)
    eeg_others=ml.drop_ID_cols(eeg_others)
    
    emg_model,eeg_model=train_models_opt(emg_others,eeg_others,args)
    
    classlabels = emg_model.classes_
    
    emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        
    targets, predlist_emg, correctness_emg, predlist_eeg, correctness_eeg, predlist_fusion, correctness_fusion = synchronously_classify(emg_ppt, eeg_ppt, emg_model, eeg_model, classlabels,args)
        
    acc_emg,acc_eeg,acc_fusion=evaluate_results(targets, predlist_emg, correctness_emg, predlist_eeg, correctness_eeg, predlist_fusion, correctness_fusion, classlabels, plot_confmats)
    
    return 1-acc_fusion

def train_bayes_fuser(model_emg,model_eeg,emg_set,eeg_set,classlabels,args):
    targets,predlist_emg,predlist_eeg,_,_,_,_=refactor_synced_predict(emg_set, eeg_set, model_emg, model_eeg, classlabels, args)
    onehot=fusion.setup_onehot(classlabels)
    onehot_pred_emg=fusion.encode_preds_onehot(predlist_emg,onehot)
    onehot_pred_eeg=fusion.encode_preds_onehot(predlist_eeg,onehot)
    fuser=fusion.train_catNB_fuser(onehot_pred_emg, onehot_pred_eeg, targets)
    return fuser, onehot

def train_svm_fuser(model_emg,model_eeg,emg_set,eeg_set,classlabels,args,sel_cols_eeg,sel_cols_emg):
    targets,predlist_emg,predlist_eeg,_,distros_emg,distros_eeg,_=refactor_synced_predict(emg_set, eeg_set, model_emg, model_eeg, classlabels, args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
    if not args['stack_distros']:
        onehot=fusion.setup_onehot(classlabels)
        onehot_pred_emg=fusion.encode_preds_onehot(predlist_emg,onehot)
        onehot_pred_eeg=fusion.encode_preds_onehot(predlist_eeg,onehot)
        fuser=fusion.train_svm_fuser(onehot_pred_emg, onehot_pred_eeg, targets, args['svmfuse'])
    else:
        onehot=None
        fuser=fusion.train_svm_fuser(distros_emg, distros_eeg, targets, args['svmfuse'])
    return fuser, onehot

def train_lda_fuser(model_emg,model_eeg,emg_set,eeg_set,classlabels,args,sel_cols_eeg,sel_cols_emg):
    targets,predlist_emg,predlist_eeg,_,distros_emg,distros_eeg,_=refactor_synced_predict(emg_set, eeg_set, model_emg, model_eeg, classlabels, args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
    if not args['stack_distros']:
        onehot=fusion.setup_onehot(classlabels)
        onehot_pred_emg=fusion.encode_preds_onehot(predlist_emg,onehot)
        onehot_pred_eeg=fusion.encode_preds_onehot(predlist_eeg,onehot)
        fuser=fusion.train_lda_fuser(onehot_pred_emg, onehot_pred_eeg, targets, args['ldafuse'])
    else:
        onehot=None
        fuser=fusion.train_lda_fuser(distros_emg, distros_eeg, targets, args['ldafuse'])
    return fuser, onehot

def train_RF_fuser(model_emg,model_eeg,emg_set,eeg_set,classlabels,args,sel_cols_eeg,sel_cols_emg):
    targets,predlist_emg,predlist_eeg,_,distros_emg,distros_eeg,_=refactor_synced_predict(emg_set, eeg_set, model_emg, model_eeg, classlabels, args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
    if not args['stack_distros']:
        onehot=fusion.setup_onehot(classlabels)
        onehot_pred_emg=fusion.encode_preds_onehot(predlist_emg,onehot)
        onehot_pred_eeg=fusion.encode_preds_onehot(predlist_eeg,onehot)
        fuser=fusion.train_rf_fuser(onehot_pred_emg, onehot_pred_eeg, targets, args['RFfuse'])
    else:
        onehot=None
        fuser=fusion.train_rf_fuser(distros_emg, distros_eeg, targets, args['RFfuse'])
    return fuser, onehot

def feature_fusion(emg_others,eeg_others,emg_ppt,eeg_ppt,args):
    '''TRAINING ON NON-PPT DATA'''
    emg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_others=emg_others.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_others=eeg_others.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    
    if emg_others['Label'].equals(eeg_others['Label']):
        #print('Target classes match, ok to merge sets')
        pass
    else:
        raise RuntimeError('Target classes should match, training sets are misaligned')
    
     
    eeg_others=ml.drop_ID_cols(eeg_others)
    emg_others=ml.drop_ID_cols(emg_others)
    
    if not args['featfuse_sel_feats_together']:
        
        if args['trialmode']=='WithinPpt':
            sel_cols_emg=feats.sel_percent_feats_df(emg_others,percent=15)
            sel_cols_emg=np.append(sel_cols_emg,emg_others.columns.get_loc('Label'))

            #sel_cols_eeg=feats.sel_percent_feats_df(eeg_others,percent=3)
            sel_cols_eeg=feats.sel_feats_l1_df(eeg_others,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
            sel_cols_eeg=np.append(sel_cols_eeg,eeg_others.columns.get_loc('Label'))
        elif args['trialmode']=='LOO':
            idx=int(emg_ppt['ID_pptID'].iloc[0])-1
            sel_cols_emg=[emg_others.columns.get_loc(col) for col in args['emg_feats_LOO'].iloc[idx].tolist()]
            sel_cols_eeg=[eeg_others.columns.get_loc(col) for col in args['eeg_feats_LOO'].iloc[idx].tolist()]
            
        emg_others = emg_others.iloc[:,sel_cols_emg] 
        eeg_others = eeg_others.iloc[:,sel_cols_eeg]


    eeg_others.drop('Label',axis='columns',inplace=True)
    eeg_others.rename(columns=lambda x: 'EEG_'+x, inplace=True)
    labelcol=emg_others.pop('Label')
    emgeeg_others=pd.concat([emg_others,eeg_others],axis=1)
    emgeeg_others['Label']=labelcol
    
    if args['featfuse_sel_feats_together']:
        if args['trialmode']=='WithinPpt':
            #sel_cols_emgeeg=feats.sel_percent_feats_df(emgeeg_others,percent=15)
            sel_cols_emgeeg=feats.sel_feats_l1_df(emgeeg_others,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats']+88)
            '''here we are taking total features = N(EEG feats) + N(EMG feats) = N(EEG) + 88'''
            sel_cols_emgeeg=np.append(sel_cols_emgeeg,emgeeg_others.columns.get_loc('Label'))
        elif args['trialmode']=='LOO':
            idx=int(emg_ppt['ID_pptID'].iloc[0])-1
            sel_cols_emgeeg=[emgeeg_others.columns.get_loc(col) for col in args['jointemgeeg_feats_LOO'].iloc[idx].tolist()]

        emgeeg_others = emgeeg_others.iloc[:,sel_cols_emgeeg]
    
    emgeeg_model = ml.train_optimise(emgeeg_others, args['featfuse']['featfuse_model_type'],args['featfuse'])
    
    classlabels = emgeeg_model.classes_
    
    '''TESTING ON PPT DATA'''
    emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
    
    index_emg_ppt=ml.pd.MultiIndex.from_arrays([emg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg_ppt=ml.pd.MultiIndex.from_arrays([eeg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_ppt=emg_ppt.loc[index_emg_ppt.isin(index_eeg_ppt)].reset_index(drop=True)
    eeg_ppt=eeg_ppt.loc[index_eeg_ppt.isin(index_emg_ppt)].reset_index(drop=True)
    
    if emg_ppt['Label'].equals(eeg_ppt['Label']):
        #print('Target classes match, ok to merge sets')
        targets=emg_ppt['Label'].values.tolist()
    else:
        raise RuntimeError('Sense check failed, target classes should match, testing sets are misaligned')
    
    eeg_ppt=ml.drop_ID_cols(eeg_ppt)
    emg_ppt=ml.drop_ID_cols(emg_ppt)
    
    if not args['featfuse_sel_feats_together']:
        '''selecting feats modally before join'''
        eeg_ppt=eeg_ppt.iloc[:,sel_cols_eeg]
        emg_ppt=emg_ppt.iloc[:,sel_cols_emg]
    
    '''joining modalities'''
    eeg_ppt.drop('Label',axis='columns',inplace=True)
    eeg_ppt.rename(columns=lambda x: 'EEG_'+x, inplace=True)
    #emg_others[('EEG_',varname)]=eeg_others[varname] for varname in eeg_others.columns.values()
    labelcol_ppt=emg_ppt.pop('Label')
    emgeeg_ppt=pd.concat([emg_ppt,eeg_ppt],axis=1)
    emgeeg_ppt['Label']=labelcol_ppt
    
    if args['featfuse_sel_feats_together']:
        '''selecting feats after join'''
        emgeeg_ppt=emgeeg_ppt.iloc[:,sel_cols_emgeeg]
        
    #emgeeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    #these are already sorted above before appending together?
    predlist_fusion=[]
        
    '''Get values from instances'''
#    IDs=list(emgeeg_ppt.filter(regex='^ID_').keys())
 #   emgeeg_vals=emgeeg_ppt.drop(IDs,axis='columns').values
    
 #   if args['featfuse_sel_feats_together']:
  #     sel_cols_emgeeg=np.append(sel_cols_emgeeg,emgeeg_ppt.columns.get_loc('Label'))
   #    emgeeg_ppt = emgeeg_ppt.iloc[:,sel_cols_emgeeg]
    
    emgeeg_vals=emgeeg_ppt.drop(['Label'],axis='columns').values
        
    '''Pass values to models'''    
    distros_fusion=ml.prob_dist(emgeeg_model,emgeeg_vals)
    for distro in distros_fusion:
        pred_fusion=ml.pred_from_distro(classlabels,distro)
        predlist_fusion.append(pred_fusion) 
    
    return targets, predlist_fusion, predlist_fusion, predlist_fusion, classlabels
    #return targets, predlist_fusion, predlist_fusion, predlist_fusion, classlabels, sel_cols_emgeeg, emgeeg_others.columns.values

def fusion_hierarchical_noCV(emg_others,eeg_others,emg_ppt,eeg_ppt,args):
    '''TRAINING ON NON-PPT DATA'''
    emg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_others=emg_others.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_others=eeg_others.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
            
    emg_others['ID_splitIndex']=emg_others['Label'].astype(str)+emg_others['ID_pptID'].astype(str)
    eeg_others['ID_splitIndex']=eeg_others['Label'].astype(str)+eeg_others['ID_pptID'].astype(str)
    #https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
    random_split=random.randint(0,100)
    emg_train_split_ML,emg_train_split_fusion=train_test_split(emg_others,test_size=0.5,random_state=random_split,stratify=emg_others[['ID_splitIndex']])
    eeg_train_split_ML,eeg_train_split_fusion=train_test_split(eeg_others,test_size=0.5,random_state=random_split,stratify=eeg_others[['ID_splitIndex']])
    #https://stackoverflow.com/questions/43095076/scikit-learn-train-test-split-can-i-ensure-same-splits-on-different-datasets
    
    emg_train_split_fusion=ml.drop_ID_cols(emg_train_split_fusion)
    
    sel_cols_emg=feats.sel_percent_feats_df(emg_train_split_fusion,percent=15)
    sel_cols_emg=np.append(sel_cols_emg,emg_train_split_fusion.columns.get_loc('Label'))
    emg_train_split_fusion=emg_train_split_fusion.iloc[:,sel_cols_emg]

    
    '''Train EEG model'''
    eeg_train_split_ML=ml.drop_ID_cols(eeg_train_split_ML)
    #sel_cols_eeg=feats.sel_percent_feats_df(eeg_train_split_ML,percent=3)
    sel_cols_eeg=feats.sel_feats_l1_df(eeg_train_split_ML,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
    sel_cols_eeg=np.append(sel_cols_eeg,eeg_train_split_ML.columns.get_loc('Label'))
    eeg_train_split_ML=eeg_train_split_ML.iloc[:,sel_cols_eeg]
    
    eeg_model = ml.train_optimise(eeg_train_split_ML, args['eeg']['eeg_model_type'], args['eeg'])
    classlabels=eeg_model.classes_
    
    '''Get values from instances'''  
    IDs=list(eeg_train_split_fusion.filter(regex='^ID_').keys())
    eeg_train_split_fusion=eeg_train_split_fusion.drop(IDs,axis='columns')
    eeg_train_split_fusion=eeg_train_split_fusion.iloc[:,sel_cols_eeg]
    eegvals=eeg_train_split_fusion.drop(['Label'],axis='columns').values

    '''Get EEG preds for EMG training'''
    eeg_preds_hierarch= []
    distros_eeg=ml.prob_dist(eeg_model,eegvals)
    for distro in distros_eeg:
        pred_eeg=ml.pred_from_distro(classlabels,distro)
        eeg_preds_hierarch.append(pred_eeg)
    
    if not args['stack_distros']:
        '''Add EEG preds to EMG training set'''
        onehot=fusion.setup_onehot(classlabels)
        onehot_pred_eeg=fusion.encode_preds_onehot(eeg_preds_hierarch,onehot)
        for idx,lab in enumerate(classlabels):
            labelcol=len(emg_train_split_fusion.columns)
            emg_train_split_fusion.insert(labelcol-1,('EEGOnehotClass'+str(lab)),onehot_pred_eeg[:,idx])
    else:
        '''Add EEG distros to EMG training set'''
        for idx,lab in enumerate(classlabels):
            labelcol=len(emg_train_split_fusion.columns)
            emg_train_split_fusion.insert(labelcol-1,('EEGProbClass'+str(lab)),distros_eeg[:,idx])   
      
    '''Train EMG model'''
    emg_model=ml.train_optimise(emg_train_split_fusion,args['emg']['emg_model_type'],args['emg'])
 
    '''-----------------'''
 
    '''TESTING ON PPT DATA'''
    emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
 
    predlist_hierarch=[]
    predlist_eeg=[]
    targets=[]
     
    index_emg=ml.pd.MultiIndex.from_arrays([emg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg=emg_ppt.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg=eeg_ppt.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    
    if emg['Label'].equals(eeg['Label']):
        targets=emg['Label'].values.tolist()
    else:
        raise Exception('Sense check failed, target label should agree between modes')

    '''Get values from instances'''
    IDs=list(emg.filter(regex='^ID_').keys())
    eeg=eeg.drop(IDs,axis='columns')
    eeg=eeg.iloc[:,sel_cols_eeg]
    eegvals=eeg.drop(['Label'],axis='columns').values    
    
    '''Get EEG Predictions'''
    distros_eeg=ml.prob_dist(eeg_model,eegvals)
    for distro in distros_eeg:
        pred_eeg=ml.pred_from_distro(classlabels,distro)
        predlist_eeg.append(pred_eeg)    
    
    emg=emg.drop(IDs,axis='columns') #drop BEFORE inserting EEGOnehot
    emg=emg.iloc[:,sel_cols_emg]
    if not args['stack_distros']:
        '''Add EEG Preds to EMG set'''
        onehot_pred_eeg=fusion.encode_preds_onehot(predlist_eeg,onehot)
        for idx,lab in enumerate(classlabels):
            labelcol=len(emg.columns)
            emg.insert(labelcol-1,('EEGOnehotClass'+str(lab)),onehot_pred_eeg[:,idx])
            #emg[('EMG1hotClass'+str(lab))]=onehot_pred_eeg[:,idx]
    else:
        '''Add EEG distros to EMG set'''
        for idx,lab in enumerate(classlabels):
            labelcol=len(emg.columns)
            emg.insert(labelcol-1,('EEGProbClass'+str(lab)),distros_eeg[:,idx])
        
    emg=emg.drop(['Label'],axis='columns')
 
    distros_emg=ml.prob_dist(emg_model,emg.values)
    for distro in distros_emg:
        pred_emg=ml.pred_from_distro(classlabels,distro)
        predlist_hierarch.append(pred_emg)
    predlist_emg=predlist_hierarch
    
    return targets, predlist_emg, predlist_eeg, predlist_hierarch, classlabels




def fusion_hierarchical(emg_others,eeg_others,emg_ppt,eeg_ppt,args):
    '''TRAINING ON NON-PPT DATA'''
    emg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_others=emg_others.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_others=eeg_others.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
            
    emg_others['ID_splitIndex']=emg_others['Label'].astype(str)+emg_others['ID_pptID'].astype(str)
    eeg_others['ID_splitIndex']=eeg_others['Label'].astype(str)+eeg_others['ID_pptID'].astype(str)
    #https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
    
    if args['trialmode']=='WithinPpt':
        sel_cols_eeg=feats.sel_feats_l1_df(ml.drop_ID_cols(eeg_others),sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
        sel_cols_eeg=np.append(sel_cols_eeg,ml.drop_ID_cols(eeg_others).columns.get_loc('Label'))
    elif args['trialmode']=='LOO':
        idx=int(eeg_ppt['ID_pptID'].iloc[0])-1
        sel_cols_eeg=[ml.drop_ID_cols(eeg_others).columns.get_loc(col) for col in args['eeg_feats_LOO'].iloc[idx].tolist()]
    
    random_split=random.randint(0,100)
    folds=StratifiedKFold(random_state=random_split,n_splits=3,shuffle=False)
    eeg_preds_hierarch= []
    for i, (index_ML, index_Fus) in enumerate(folds.split(eeg_others,eeg_others['ID_splitIndex'])):
        eeg_train_split_ML=eeg_others.iloc[index_ML]
        eeg_train_split_fusion=eeg_others.iloc[index_Fus]
        
        '''Train EEG model'''
        eeg_train_split_ML=ml.drop_ID_cols(eeg_train_split_ML)
        #sel_cols_eeg=feats.sel_percent_feats_df(eeg_train_split_ML,percent=3)
        eeg_train_split_ML=eeg_train_split_ML.iloc[:,sel_cols_eeg]
        
        eeg_model = ml.train_optimise(eeg_train_split_ML, args['eeg']['eeg_model_type'], args['eeg'])
        classlabels=eeg_model.classes_
        
        '''Get values from instances'''  
        IDs=list(eeg_train_split_fusion.filter(regex='^ID_').keys())
        eeg_train_split_fusion=eeg_train_split_fusion.drop(IDs,axis='columns')
        eeg_train_split_fusion=eeg_train_split_fusion.iloc[:,sel_cols_eeg]
        eegvals=eeg_train_split_fusion.drop(['Label'],axis='columns').values
    
        '''Get EEG preds for EMG training'''
        
        distros_eeg=ml.prob_dist(eeg_model,eegvals)
        if not args['stack_distros']:
            predlist=[]
            for distro in distros_eeg:
                predlist.append(ml.pred_from_distro(classlabels,distro))
            onehot=fusion.setup_onehot(classlabels)
            onehot_pred_eeg=fusion.encode_preds_onehot(eeg_preds_hierarch,onehot)
            preds=ml.pd.DataFrame(onehot_pred_eeg,index=index_Fus,columns=[('EEGOnehotClass'+str(col)) for col in classlabels])
        else:
            preds=ml.pd.DataFrame(distros_eeg,index=index_Fus,columns=[('EEGProbClass'+str(col)) for col in classlabels])
        if len(eeg_preds_hierarch)==0:
            eeg_preds_hierarch=preds
        else:
            eeg_preds_hierarch=ml.pd.concat([eeg_preds_hierarch,preds],axis=0)
    
    eeg_others=ml.drop_ID_cols(eeg_others)
    eeg_others=eeg_others.iloc[:,sel_cols_eeg]
    eeg_model=ml.train_optimise(eeg_others, args['eeg']['eeg_model_type'], args['eeg'])
    classlabels=eeg_model.classes_
    
    emg_train=emg_others
    emg_train=ml.drop_ID_cols(emg_train)
    if args['trialmode']=='WithinPpt':
        sel_cols_emg=feats.sel_percent_feats_df(emg_train,percent=15)
        sel_cols_emg=np.append(sel_cols_emg,emg_train.columns.get_loc('Label'))
    elif args['trialmode']=='LOO':
        idx=int(emg_ppt['ID_pptID'].iloc[0])-1
        sel_cols_emg=[emg_train.columns.get_loc(col) for col in args['emg_feats_LOO'].iloc[idx].tolist()]
    emg_train=emg_train.iloc[:,sel_cols_emg]
    
    lab=emg_train.pop('Label')
    emg_train=ml.pd.concat([emg_train,eeg_preds_hierarch],axis=1)
    emg_train['Label']=lab  
      
    '''Train EMG model'''
    emg_model=ml.train_optimise(emg_train,args['emg']['emg_model_type'],args['emg'])
 
    '''-----------------'''
 
    '''TESTING ON PPT DATA'''
    emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
 
    predlist_hierarch=[]
    predlist_eeg=[]
    targets=[]
     
    index_emg=ml.pd.MultiIndex.from_arrays([emg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg=emg_ppt.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg=eeg_ppt.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    
    if emg['Label'].equals(eeg['Label']):
        targets=emg['Label'].values.tolist()
    else:
        raise Exception('Sense check failed, target label should agree between modes')

    '''Get values from instances'''
    IDs=list(emg.filter(regex='^ID_').keys())
    eeg=eeg.drop(IDs,axis='columns')
    eeg=eeg.iloc[:,sel_cols_eeg]
    eegvals=eeg.drop(['Label'],axis='columns').values    
    
    '''Get EEG Predictions'''
    distros_eeg=ml.prob_dist(eeg_model,eegvals)
    for distro in distros_eeg:
        pred_eeg=ml.pred_from_distro(classlabels,distro)
        predlist_eeg.append(pred_eeg)    
    
    emg=emg.drop(IDs,axis='columns') #drop BEFORE inserting EEGOnehot
    emg=emg.iloc[:,sel_cols_emg]
    if not args['stack_distros']:
        '''Add EEG Preds to EMG set'''
        onehot_pred_eeg=fusion.encode_preds_onehot(predlist_eeg,onehot)
        for idx,lab in enumerate(classlabels):
            labelcol=len(emg.columns)
            emg.insert(labelcol-1,('EEGOnehotClass'+str(lab)),onehot_pred_eeg[:,idx])
            #emg[('EMG1hotClass'+str(lab))]=onehot_pred_eeg[:,idx]
    else:
        '''Add EEG distros to EMG set'''
        for idx,lab in enumerate(classlabels):
            labelcol=len(emg.columns)
            emg.insert(labelcol-1,('EEGProbClass'+str(lab)),distros_eeg[:,idx])
        
    emg=emg.drop(['Label'],axis='columns')
 
    distros_emg=ml.prob_dist(emg_model,emg.values)
    for distro in distros_emg:
        pred_emg=ml.pred_from_distro(classlabels,distro)
        predlist_hierarch.append(pred_emg)
    predlist_emg=predlist_hierarch
    
    return targets, predlist_emg, predlist_eeg, predlist_hierarch, classlabels


def fusion_hierarchical_inv(emg_others,eeg_others,emg_ppt,eeg_ppt,args):
    '''TRAINING ON NON-PPT DATA'''
    emg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_others=emg_others.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_others=eeg_others.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
            
    emg_others['ID_splitIndex']=emg_others['Label'].astype(str)+emg_others['ID_pptID'].astype(str)
    eeg_others['ID_splitIndex']=eeg_others['Label'].astype(str)+eeg_others['ID_pptID'].astype(str)
    #https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
    
    if args['trialmode']=='WithinPpt':
        sel_cols_emg=feats.sel_percent_feats_df(ml.drop_ID_cols(emg_others),percent=15)
        sel_cols_emg=np.append(sel_cols_emg,ml.drop_ID_cols(emg_others).columns.get_loc('Label'))
    elif args['trialmode']=='LOO':
        idx=int(emg_ppt['ID_pptID'].iloc[0])-1
        sel_cols_emg=[ml.drop_ID_cols(emg_others).columns.get_loc(col) for col in args['emg_feats_LOO'].iloc[idx].tolist()]
	
    random_split=random.randint(0,100)
    folds=StratifiedKFold(random_state=random_split,n_splits=3,shuffle=False)
    emg_preds_hierarch= []
    for i, (index_ML, index_Fus) in enumerate(folds.split(emg_others,emg_others['ID_splitIndex'])):
	
        emg_train_split_ML=emg_others.iloc[index_ML]
        emg_train_split_fusion=emg_others.iloc[index_Fus]
		
        '''Train EMG model'''
        emg_train_split_ML=ml.drop_ID_cols(emg_train_split_ML)
        emg_train_split_ML=emg_train_split_ML.iloc[:,sel_cols_emg]
    
        emg_model = ml.train_optimise(emg_train_split_ML, args['emg']['emg_model_type'], args['emg'])
        classlabels=emg_model.classes_
    
        '''Get values from instances'''
        IDs=list(emg_train_split_fusion.filter(regex='^ID_').keys())
        emg_train_split_fusion=emg_train_split_fusion.drop(IDs,axis='columns')
        emg_train_split_fusion=emg_train_split_fusion.iloc[:,sel_cols_emg]
        emgvals=emg_train_split_fusion.drop(['Label'],axis='columns').values

    
        '''Get EMG preds for EEG training'''
        distros_emg=ml.prob_dist(emg_model,emgvals)
        if not args['stack_distros']:    
            '''Add EMG preds to EEG training set'''
            predlist=[]
            for distro in distros_emg:
                predlist.append(ml.pred_from_distro(classlabels,distro))
            onehot=fusion.setup_onehot(classlabels)
            onehot_pred_emg=fusion.encode_preds_onehot(emg_preds_hierarch,onehot)
            preds=ml.pd.DataFrame(onehot_pred_emg,index=index_Fus,columns=[('EMGOnehotClass'+str(col)) for col in classlabels])
        else:
            '''Add EMG distros to EEG training set'''
            preds=ml.pd.DataFrame(distros_emg,index=index_Fus,columns=[('EMGProbClass'+str(col)) for col in classlabels])
        if len(emg_preds_hierarch)==0:
            emg_preds_hierarch=preds
        else:
            emg_preds_hierarch=ml.pd.concat([emg_preds_hierarch,preds],axis=0)
	
    emg_others=ml.drop_ID_cols(emg_others)
    emg_others=emg_others.iloc[:,sel_cols_emg]
    emg_model=ml.train_optimise(emg_others,args['emg']['emg_model_type'],args['emg'])
    classlabels=emg_model.classes_
	
    eeg_train=eeg_others
    eeg_train=ml.drop_ID_cols(eeg_train)
    if args['trialmode']=='WithinPpt':
        sel_cols_eeg=feats.sel_feats_l1_df(eeg_train,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
        sel_cols_eeg=np.append(sel_cols_eeg,eeg_train.columns.get_loc('Label'))
    elif args['trialmode']=='LOO':
        idx=int(eeg_ppt['ID_pptID'].iloc[0])-1
        sel_cols_eeg=[eeg_train.columns.get_loc(col) for col in args['eeg_feats_LOO'].iloc[idx].tolist()]
    eeg_train=eeg_train.iloc[:,sel_cols_eeg]
	
    lab=eeg_train.pop('Label')
    eeg_train=ml.pd.concat([eeg_train,emg_preds_hierarch],axis=1)
    eeg_train['Label']=lab  
	
    '''Train EEG model'''
    eeg_model=ml.train_optimise(eeg_train,args['eeg']['eeg_model_type'],args['eeg'])
 
    '''-----------------'''
 
    '''TESTING ON PPT DATA'''
    emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
 
    predlist_hierarch=[]
    predlist_emg=[]
    targets=[]
     
    index_emg=ml.pd.MultiIndex.from_arrays([emg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
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
    emgvals=emg.drop(['Label'],axis='columns').values    
    
   
    '''Get EMG Predictions'''
    distros_emg=ml.prob_dist(emg_model,emgvals)
    for distro in distros_emg:
        pred_emg=ml.pred_from_distro(classlabels,distro)
        predlist_emg.append(pred_emg)
    
    eeg=eeg.drop(IDs,axis='columns') #drop BEFORE inserting EMGOnehot
    eeg=eeg.iloc[:,sel_cols_eeg]
    if not args['stack_distros']:     
        '''Add EMG Preds to EEG set'''
        onehot_pred_emg=fusion.encode_preds_onehot(predlist_emg,onehot)
        for idx,lab in enumerate(classlabels):
            labelcol=len(eeg.columns)
            eeg.insert(labelcol-1,('EMGOnehotClass'+str(lab)),onehot_pred_emg[:,idx])
    else:
        '''Add EMG distros to EEG set'''
        for idx,lab in enumerate(classlabels):
            labelcol=len(eeg.columns)
            eeg.insert(labelcol-1,('EMGProbClass'+str(lab)),distros_emg[:,idx])
    
    eeg=eeg.drop(['Label'],axis='columns')
 
    distros_eeg=ml.prob_dist(eeg_model,eeg.values)
    for distro in distros_eeg:
        pred_eeg=ml.pred_from_distro(classlabels,distro)
        predlist_hierarch.append(pred_eeg)
    predlist_eeg=predlist_hierarch
    
    if args['get_train_acc']:
        predlist_emgtrain=[]  
        predlist_fustrain=[]        
        eeg_train=eeg_others.drop(IDs,axis='columns')
        eeg_train=eeg_train.iloc[:,sel_cols_eeg]
        emg_train=emg_others.drop(IDs,axis='columns')
        emg_train=emg_train.iloc[:,sel_cols_emg]
        traintargs=eeg_train['Label'].values.tolist()
        emgtrainvals=emg_train.drop('Label',axis='columns')#does this need to be .values?
        distros_emgtrain=ml.prob_dist(emg_model,emgtrainvals)
        for distro in distros_emgtrain:
            pred_emgtrain=ml.pred_from_distro(classlabels,distro)
            predlist_emgtrain.append(pred_emgtrain)
        onehot_pred_emgtrain=fusion.encode_preds_onehot(predlist_emgtrain,onehot)
        for idx,lab in enumerate(classlabels):
            labelcol=len(eeg_train.columns)
            eeg_train.insert(labelcol-1,('EMGOnehotClass'+str(lab)),onehot_pred_emgtrain[:,idx])
        eeg_train=eeg_train.drop(['Label'],axis='columns')
        distros_eegtrain=ml.prob_dist(eeg_model,eeg_train.values)
        for distro in distros_eegtrain:
            pred_eegtrain=ml.pred_from_distro(classlabels,distro)
            predlist_fustrain.append(pred_eegtrain)
        return targets, predlist_emg, predlist_eeg, predlist_hierarch, classlabels, traintargs, predlist_fustrain
   
    else:    
        return targets, predlist_emg, predlist_eeg, predlist_hierarch, classlabels


def fusion_hierarchical_inv_noCV(emg_others,eeg_others,emg_ppt,eeg_ppt,args):
    '''TRAINING ON NON-PPT DATA'''
    emg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_others=emg_others.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_others=eeg_others.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
            
    emg_others['ID_splitIndex']=emg_others['Label'].astype(str)+emg_others['ID_pptID'].astype(str)
    eeg_others['ID_splitIndex']=eeg_others['Label'].astype(str)+eeg_others['ID_pptID'].astype(str)
    #https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
    random_split=random.randint(0,100)
    emg_train_split_ML,emg_train_split_fusion=train_test_split(emg_others,test_size=0.5,random_state=random_split,stratify=emg_others[['ID_splitIndex']])
    eeg_train_split_ML,eeg_train_split_fusion=train_test_split(eeg_others,test_size=0.5,random_state=random_split,stratify=eeg_others[['ID_splitIndex']])
    #https://stackoverflow.com/questions/43095076/scikit-learn-train-test-split-can-i-ensure-same-splits-on-different-datasets
    
    eeg_train_split_fusion=ml.drop_ID_cols(eeg_train_split_fusion)
    
    #sel_cols_eeg=feats.sel_percent_feats_df(eeg_train_split_fusion,percent=3)
    sel_cols_eeg=feats.sel_feats_l1_df(eeg_train_split_fusion,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
    sel_cols_eeg=np.append(sel_cols_eeg,eeg_train_split_fusion.columns.get_loc('Label'))
    eeg_train_split_fusion=eeg_train_split_fusion.iloc[:,sel_cols_eeg]
    
    '''Train EMG model'''
    emg_train_split_ML=ml.drop_ID_cols(emg_train_split_ML)
    sel_cols_emg=feats.sel_percent_feats_df(emg_train_split_ML,percent=15)
    sel_cols_emg=np.append(sel_cols_emg,emg_train_split_ML.columns.get_loc('Label'))
    emg_train_split_ML=emg_train_split_ML.iloc[:,sel_cols_emg]
    
    emg_model = ml.train_optimise(emg_train_split_ML, args['emg']['emg_model_type'], args['emg'])
    classlabels=emg_model.classes_
    
    '''Get values from instances'''
    IDs=list(emg_train_split_fusion.filter(regex='^ID_').keys())
    emg_train_split_fusion=emg_train_split_fusion.drop(IDs,axis='columns')
    emg_train_split_fusion=emg_train_split_fusion.iloc[:,sel_cols_emg]
    emgvals=emg_train_split_fusion.drop(['Label'],axis='columns').values

    
    '''Get EMG preds for EEG training'''
    emg_preds_hierarch= []
    distros_emg=ml.prob_dist(emg_model,emgvals)
    for distro in distros_emg:
        pred_emg=ml.pred_from_distro(classlabels,distro)
        emg_preds_hierarch.append(pred_emg)
        
    if not args['stack_distros']:    
        '''Add EMG preds to EEG training set'''
        onehot=fusion.setup_onehot(classlabels)
        onehot_pred_emg=fusion.encode_preds_onehot(emg_preds_hierarch,onehot)
        for idx,lab in enumerate(classlabels):
            labelcol=len(eeg_train_split_fusion.columns)
            eeg_train_split_fusion.insert(labelcol-1,('EMG1hotClass'+str(lab)),onehot_pred_emg[:,idx])
    else:
        '''Add EMG distros to EEG training set'''
        for idx,lab in enumerate(classlabels):
            labelcol=len(eeg_train_split_fusion.columns)
            eeg_train_split_fusion.insert(labelcol-1,('EMGProbClass'+str(lab)),distros_emg[:,idx])
    
    '''Train EEG model'''
    eeg_model=ml.train_optimise(eeg_train_split_fusion,args['eeg']['eeg_model_type'],args['eeg'])
 
    '''-----------------'''
 
    '''TESTING ON PPT DATA'''
    emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
 
    predlist_hierarch=[]
    predlist_emg=[]
    targets=[]
     
    index_emg=ml.pd.MultiIndex.from_arrays([emg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
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
    emgvals=emg.drop(['Label'],axis='columns').values    
    
   
    '''Get EMG Predictions'''
    distros_emg=ml.prob_dist(emg_model,emgvals)
    for distro in distros_emg:
        pred_emg=ml.pred_from_distro(classlabels,distro)
        predlist_emg.append(pred_emg)
    
    eeg=eeg.drop(IDs,axis='columns') #drop BEFORE inserting EMGOnehot
    eeg=eeg.iloc[:,sel_cols_eeg]
    if not args['stack_distros']:     
        '''Add EMG Preds to EEG set'''
        onehot_pred_emg=fusion.encode_preds_onehot(predlist_emg,onehot)
        for idx,lab in enumerate(classlabels):
            labelcol=len(eeg.columns)
            eeg.insert(labelcol-1,('EMGOnehotClass'+str(lab)),onehot_pred_emg[:,idx])
    else:
        '''Add EMG distros to EEG set'''
        for idx,lab in enumerate(classlabels):
            labelcol=len(eeg.columns)
            eeg.insert(labelcol-1,('EMGProbClass'+str(lab)),distros_emg[:,idx])
    
    eeg=eeg.drop(['Label'],axis='columns')
 
    distros_eeg=ml.prob_dist(eeg_model,eeg.values)
    for distro in distros_eeg:
        pred_eeg=ml.pred_from_distro(classlabels,distro)
        predlist_hierarch.append(pred_eeg)
    predlist_eeg=predlist_hierarch
    
    if args['get_train_acc']:
        predlist_emgtrain=[]  
        predlist_fustrain=[]        
        #eeg_train=ml.drop_ID_cols(eeg_others)
        eeg_train=eeg_others.drop(IDs,axis='columns')
        eeg_train=eeg_train.iloc[:,sel_cols_eeg]
        #emg_train=ml.drop_ID_cols(emg_others)
        emg_train=emg_others.drop(IDs,axis='columns')
        emg_train=emg_train.iloc[:,sel_cols_emg]
        traintargs=eeg_train['Label'].values.tolist()
        emgtrainvals=emg_train.drop('Label',axis='columns')#does this need to be .values?
        distros_emgtrain=ml.prob_dist(emg_model,emgtrainvals)
        for distro in distros_emgtrain:
            pred_emgtrain=ml.pred_from_distro(classlabels,distro)
            predlist_emgtrain.append(pred_emgtrain)
        onehot_pred_emgtrain=fusion.encode_preds_onehot(predlist_emgtrain,onehot)
        for idx,lab in enumerate(classlabels):
            labelcol=len(eeg_train.columns)
            eeg_train.insert(labelcol-1,('EMGOnehotClass'+str(lab)),onehot_pred_emgtrain[:,idx])
        eeg_train=eeg_train.drop(['Label'],axis='columns')
        distros_eegtrain=ml.prob_dist(eeg_model,eeg_train.values)
        for distro in distros_eegtrain:
            pred_eegtrain=ml.pred_from_distro(classlabels,distro)
            predlist_fustrain.append(pred_eegtrain)
        return targets, predlist_emg, predlist_eeg, predlist_hierarch, classlabels, traintargs, predlist_fustrain
   
    else:    
        return targets, predlist_emg, predlist_eeg, predlist_hierarch, classlabels


def only_EMG(emg_others,eeg_others,emg_ppt,eeg_ppt,args):
    '''TRAINING ON NON-PPT DATA'''
    emg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_others=emg_others.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_others=eeg_others.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    

    '''Train EMG model'''
    emg_train=ml.drop_ID_cols(emg_others)
    if args['trialmode']=='WithinPpt':
        sel_cols_emg=feats.sel_percent_feats_df(emg_train,percent=15)
        sel_cols_emg=np.append(sel_cols_emg,emg_train.columns.get_loc('Label'))
    elif args['trialmode']=='LOO':
        idx=int(emg_ppt['ID_pptID'].iloc[0])-1
        sel_cols_emg=[emg_train.columns.get_loc(col) for col in args['emg_feats_LOO'].iloc[idx].tolist()]
    emg_train=emg_train.iloc[:,sel_cols_emg]
    
    emg_model = ml.train_optimise(emg_train, args['emg']['emg_model_type'], args['emg'])
    classlabels=emg_model.classes_   
 
 
    '''TESTING ON PPT DATA'''
    emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
 
    predlist_emg=[]
    targets=[]
     
    index_emg=ml.pd.MultiIndex.from_arrays([emg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
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
    emgvals=emg.drop(['Label'],axis='columns').values    
    
    '''Get EMG Predictions'''
    distros_emg=ml.prob_dist(emg_model,emgvals)
    for distro in distros_emg:
        pred_emg=ml.pred_from_distro(classlabels,distro)
        predlist_emg.append(pred_emg)
    
    if args['get_train_acc']:
        predlist_emgtrain=[]
        traintargs=emg_train['Label'].values.tolist()
        emgtrainvals=emg_train.drop('Label',axis='columns') #why DOESNT this need to be .values?
        distros_emgtrain=ml.prob_dist(emg_model,emgtrainvals)
        for distro in distros_emgtrain:
            pred_emgtrain=ml.pred_from_distro(classlabels,distro)
            predlist_emgtrain.append(pred_emgtrain)
        return targets, predlist_emg, predlist_emg, predlist_emg, classlabels, traintargs, predlist_emgtrain
   
    else:
        return targets, predlist_emg, predlist_emg, predlist_emg, classlabels


def only_EEG(emg_others,eeg_others,emg_ppt,eeg_ppt,args):
    '''TRAINING ON NON-PPT DATA'''
    emg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_others=emg_others.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_others=eeg_others.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    

    '''Train EEG model'''
    eeg_train=ml.drop_ID_cols(eeg_others)
    if args['trialmode']=='WithinPpt':
        #sel_cols_eeg=feats.sel_percent_feats_df(eeg_train,percent=3)
        sel_cols_eeg=feats.sel_feats_l1_df(eeg_train,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
        #print('reduced to '+str(len(sel_cols_eeg))+' cols (line 944)')
        sel_cols_eeg=np.append(sel_cols_eeg,eeg_train.columns.get_loc('Label'))
    elif args['trialmode']=='LOO':
        idx=int(eeg_ppt['ID_pptID'].iloc[0])-1
        sel_cols_eeg=[eeg_train.columns.get_loc(col) for col in args['eeg_feats_LOO'].iloc[idx].tolist()]
    eeg_train=eeg_train.iloc[:,sel_cols_eeg]
    
    eeg_model = ml.train_optimise(eeg_train, args['eeg']['eeg_model_type'], args['eeg'],args['bag_eeg'])
    classlabels=eeg_model.classes_   
 
 
    '''TESTING ON PPT DATA'''
    emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
 
    predlist_eeg=[]
    targets=[]
     
    index_emg=ml.pd.MultiIndex.from_arrays([emg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg=emg_ppt.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg=eeg_ppt.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    
    if eeg['Label'].equals(emg['Label']):
        targets=eeg['Label'].values.tolist()
    else:
        raise Exception('Sense check failed, target label should agree between modes')        
     
    '''Get values from instances'''
    IDs=list(emg.filter(regex='^ID_').keys())
    eeg=eeg.drop(IDs,axis='columns')
    eeg=eeg.iloc[:,sel_cols_eeg]
    eegvals=eeg.drop(['Label'],axis='columns').values    
    
    '''Get EEG Predictions'''
    distros_eeg=ml.prob_dist(eeg_model,eegvals)
    for distro in distros_eeg:
        pred_eeg=ml.pred_from_distro(classlabels,distro)
        predlist_eeg.append(pred_eeg)
        
    if args['get_train_acc']:
        predlist_eegtrain=[]
        traintargs=eeg_train['Label'].values.tolist()
        eegtrainvals=eeg_train.drop('Label',axis='columns') #why DOESNT this need to be .values?
        distros_eegtrain=ml.prob_dist(eeg_model,eegtrainvals)
        for distro in distros_eegtrain:
            pred_eegtrain=ml.pred_from_distro(classlabels,distro)
            predlist_eegtrain.append(pred_eegtrain)
        return targets, predlist_eeg, predlist_eeg, predlist_eeg, classlabels, traintargs, predlist_eegtrain
   
    else:
        return targets, predlist_eeg, predlist_eeg, predlist_eeg, classlabels



def fusion_SVM(emg_train, eeg_train, emg_test, eeg_test, args):
    emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_train=emg_train.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_train=eeg_train.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
            
    emg_train['ID_splitIndex']=emg_train['Label'].astype(str)+emg_train['ID_pptID'].astype(str)
    eeg_train['ID_splitIndex']=eeg_train['Label'].astype(str)+eeg_train['ID_pptID'].astype(str)
    #https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
    
    if args['trialmode']=='WithinPpt':
        sel_cols_emg=feats.sel_percent_feats_df(ml.drop_ID_cols(emg_train),percent=15)
        sel_cols_emg=np.append(sel_cols_emg,ml.drop_ID_cols(emg_train).columns.get_loc('Label'))
        sel_cols_eeg=feats.sel_feats_l1_df(ml.drop_ID_cols(eeg_train),sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
        sel_cols_eeg=np.append(sel_cols_eeg,ml.drop_ID_cols(eeg_train).columns.get_loc('Label'))
    elif args['trialmode']=='LOO':
        idx=int(eeg_test['ID_pptID'].iloc[0])-1
        sel_cols_eeg=[ml.drop_ID_cols(eeg_train).columns.get_loc(col) for col in args['eeg_feats_LOO'].iloc[idx].tolist()]
        sel_cols_emg=[ml.drop_ID_cols(emg_train).columns.get_loc(col) for col in args['emg_feats_LOO'].iloc[idx].tolist()]
    
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
        
        emg_model,eeg_model=train_models_opt(emg_train_split_ML,eeg_train_split_ML,args)    
        classlabels = emg_model.classes_
        
        targets,predlist_emg,predlist_eeg,_,distros_emg,distros_eeg,_=refactor_synced_predict(emg_train_split_fusion, eeg_train_split_fusion, emg_model, eeg_model, classlabels, args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
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
    emg_model,eeg_model=train_models_opt(emg_train,eeg_train,args)
    
    if not args['stack_distros']:
        onehotEncoder=fusion.setup_onehot(classlabels)
        onehot_pred_emg=fusion.encode_preds_onehot(fusdistros_emg,onehotEncoder)
        onehot_pred_eeg=fusion.encode_preds_onehot(fusdistros_eeg,onehotEncoder)
        fuser=fusion.train_svm_fuser(onehot_pred_emg, onehot_pred_eeg, fustargets, args['svmfuse'])
    else:
        onehotEncoder=None
        fuser=fusion.train_svm_fuser(fusdistros_emg, fusdistros_eeg, fustargets, args['svmfuse'])
    
    '---------------'
    emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
    targets, predlist_emg, predlist_eeg, _, distros_emg, distros_eeg, _  = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
    
    if args['stack_distros']:
        predlist_fusion=fusion.svm_fusion(fuser,onehotEncoder,distros_emg,distros_eeg,classlabels)
    else:
        predlist_fusion=fusion.svm_fusion(fuser,onehotEncoder,predlist_emg,predlist_eeg,classlabels)
    
    if args['get_train_acc']:
        emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
        traintargs, predlist_emgtrain, predlist_eegtrain, _, distros_emgtrain, distros_eegtrain, _ = refactor_synced_predict(emg_train, eeg_train, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
        if args['stack_distros']:
            predlist_train=fusion.svm_fusion(fuser,onehotEncoder,distros_emgtrain,distros_eegtrain,classlabels)
        else:
            predlist_train=fusion.svm_fusion(fuser,onehotEncoder,predlist_emgtrain,predlist_eegtrain,classlabels)
        return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train  
    else:
        return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels



def fusion_SVM_noCV(emg_train, eeg_train, emg_test, eeg_test, args):
    emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_train=emg_train.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_train=eeg_train.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
            
    emg_train['ID_splitIndex']=emg_train['Label'].astype(str)+emg_train['ID_pptID'].astype(str)
    eeg_train['ID_splitIndex']=eeg_train['Label'].astype(str)+eeg_train['ID_pptID'].astype(str)
    #https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
    random_split=random.randint(0,100)
    emg_train_split_ML,emg_train_split_fusion=train_test_split(emg_train,test_size=0.33,random_state=random_split,stratify=emg_train[['ID_splitIndex']])
    eeg_train_split_ML,eeg_train_split_fusion=train_test_split(eeg_train,test_size=0.33,random_state=random_split,stratify=eeg_train[['ID_splitIndex']])
    #https://stackoverflow.com/questions/43095076/scikit-learn-train-test-split-can-i-ensure-same-splits-on-different-datasets
    
    
    if args['scalingtype']:
            emg_train_split_ML,emgscaler=feats.scale_feats_train(emg_train_split_ML,args['scalingtype'])
            eeg_train_split_ML,eegscaler=feats.scale_feats_train(eeg_train_split_ML,args['scalingtype'])
            emg_train_split_fusion=feats.scale_feats_test(emg_train_split_fusion,emgscaler)
            eeg_train_split_fusion=feats.scale_feats_test(eeg_train_split_fusion,eegscaler)
            emg_test=feats.scale_feats_test(emg_test,emgscaler)
            eeg_test=feats.scale_feats_test(eeg_test,eegscaler)


    emg_train_split_ML=ml.drop_ID_cols(emg_train_split_ML)
    eeg_train_split_ML=ml.drop_ID_cols(eeg_train_split_ML)
    
    
    sel_cols_emg=feats.sel_percent_feats_df(emg_train_split_ML,percent=15)
    sel_cols_emg=np.append(sel_cols_emg,emg_train_split_ML.columns.get_loc('Label'))
    emg_train_split_ML=emg_train_split_ML.iloc[:,sel_cols_emg]
    
    #sel_cols_eeg=feats.sel_percent_feats_df(eeg_train_split_ML,percent=3)
    sel_cols_eeg=feats.sel_feats_l1_df(eeg_train_split_ML,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
    sel_cols_eeg=np.append(sel_cols_eeg,eeg_train_split_ML.columns.get_loc('Label'))
    eeg_train_split_ML=eeg_train_split_ML.iloc[:,sel_cols_eeg]
       
    
    emg_model,eeg_model=train_models_opt(emg_train_split_ML,eeg_train_split_ML,args)
    
    classlabels = emg_model.classes_

    emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
    targets, predlist_emg, predlist_eeg, _, distros_emg, distros_eeg, _  = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
    
    fuser,onehotEncoder=train_svm_fuser(emg_model,eeg_model,emg_train_split_fusion,eeg_train_split_fusion,classlabels,args,sel_cols_eeg,sel_cols_emg)
    if args['stack_distros']:
        predlist_fusion=fusion.svm_fusion(fuser,onehotEncoder,distros_emg,distros_eeg,classlabels)
    else:
        predlist_fusion=fusion.svm_fusion(fuser,onehotEncoder,predlist_emg,predlist_eeg,classlabels)
    
    if args['get_train_acc']:
        emg_train=feats.scale_feats_test(emg_train,emgscaler)
        eeg_train=feats.scale_feats_test(eeg_train,eegscaler)
        emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
        traintargs, predlist_emgtrain, predlist_eegtrain, _, distros_emgtrain, distros_eegtrain, _ = refactor_synced_predict(emg_train, eeg_train, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
        if args['stack_distros']:
            predlist_train=fusion.svm_fusion(fuser,onehotEncoder,distros_emgtrain,distros_eegtrain,classlabels)
        else:
            predlist_train=fusion.svm_fusion(fuser,onehotEncoder,predlist_emgtrain,predlist_eegtrain,classlabels)
        return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train  
    else:
        return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels



def fusion_LDA(emg_train, eeg_train, emg_test, eeg_test, args):
    emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_train=emg_train.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_train=eeg_train.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
            
    emg_train['ID_splitIndex']=emg_train['Label'].astype(str)+emg_train['ID_pptID'].astype(str)
    eeg_train['ID_splitIndex']=eeg_train['Label'].astype(str)+eeg_train['ID_pptID'].astype(str)
    #https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
    
    if args['trialmode']=='WithinPpt':
        sel_cols_emg=feats.sel_percent_feats_df(ml.drop_ID_cols(emg_train),percent=15)
        sel_cols_emg=np.append(sel_cols_emg,ml.drop_ID_cols(emg_train).columns.get_loc('Label'))
        sel_cols_eeg=feats.sel_feats_l1_df(ml.drop_ID_cols(eeg_train),sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
        sel_cols_eeg=np.append(sel_cols_eeg,ml.drop_ID_cols(eeg_train).columns.get_loc('Label'))
    elif args['trialmode']=='LOO':
        idx=int(eeg_test['ID_pptID'].iloc[0])-1
        sel_cols_eeg=[ml.drop_ID_cols(eeg_train).columns.get_loc(col) for col in args['eeg_feats_LOO'].iloc[idx].tolist()]
        sel_cols_emg=[ml.drop_ID_cols(emg_train).columns.get_loc(col) for col in args['emg_feats_LOO'].iloc[idx].tolist()]
    
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
        
        emg_model,eeg_model=train_models_opt(emg_train_split_ML,eeg_train_split_ML,args)    
        classlabels = emg_model.classes_
        
        targets,predlist_emg,predlist_eeg,_,distros_emg,distros_eeg,_=refactor_synced_predict(emg_train_split_fusion, eeg_train_split_fusion, emg_model, eeg_model, classlabels, args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
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
    emg_model,eeg_model=train_models_opt(emg_train,eeg_train,args)
    
    if not args['stack_distros']:
        onehotEncoder=fusion.setup_onehot(classlabels)
        onehot_pred_emg=fusion.encode_preds_onehot(fusdistros_emg,onehotEncoder)
        onehot_pred_eeg=fusion.encode_preds_onehot(fusdistros_eeg,onehotEncoder)
        fuser=fusion.train_lda_fuser(onehot_pred_emg, onehot_pred_eeg, fustargets, args['ldafuse'])
    else:
        onehotEncoder=None
        fuser=fusion.train_lda_fuser(fusdistros_emg, fusdistros_eeg, fustargets, args['ldafuse'])
    
    '---------------'
    emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
    targets, predlist_emg, predlist_eeg, _, distros_emg, distros_eeg, _  = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
    
    if args['stack_distros']:
        predlist_fusion=fusion.lda_fusion(fuser,onehotEncoder,distros_emg,distros_eeg,classlabels)
    else:
        predlist_fusion=fusion.lda_fusion(fuser,onehotEncoder,predlist_emg,predlist_eeg,classlabels)
    
    if args['get_train_acc']:
        emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
        traintargs, predlist_emgtrain, predlist_eegtrain, _, distros_emgtrain, distros_eegtrain, _ = refactor_synced_predict(emg_train, eeg_train, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
        if args['stack_distros']:
            predlist_train=fusion.lda_fusion(fuser,onehotEncoder,distros_emgtrain,distros_eegtrain,classlabels)
        else:
            predlist_train=fusion.lda_fusion(fuser,onehotEncoder,predlist_emgtrain,predlist_eegtrain,classlabels)
        return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train  
    else:
        return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels



def fusion_LDA_noCV(emg_train, eeg_train, emg_test, eeg_test, args):
    emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_train=emg_train.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_train=eeg_train.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
            
    emg_train['ID_splitIndex']=emg_train['Label'].astype(str)+emg_train['ID_pptID'].astype(str)
    eeg_train['ID_splitIndex']=eeg_train['Label'].astype(str)+eeg_train['ID_pptID'].astype(str)
    #https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
    random_split=random.randint(0,100)
    emg_train_split_ML,emg_train_split_fusion=train_test_split(emg_train,test_size=0.33,random_state=random_split,stratify=emg_train[['ID_splitIndex']])
    eeg_train_split_ML,eeg_train_split_fusion=train_test_split(eeg_train,test_size=0.33,random_state=random_split,stratify=eeg_train[['ID_splitIndex']])
    #https://stackoverflow.com/questions/43095076/scikit-learn-train-test-split-can-i-ensure-same-splits-on-different-datasets
    
    
    if args['scalingtype']:
            emg_train_split_ML,emgscaler=feats.scale_feats_train(emg_train_split_ML,args['scalingtype'])
            eeg_train_split_ML,eegscaler=feats.scale_feats_train(eeg_train_split_ML,args['scalingtype'])
            emg_train_split_fusion=feats.scale_feats_test(emg_train_split_fusion,emgscaler)
            eeg_train_split_fusion=feats.scale_feats_test(eeg_train_split_fusion,eegscaler)
            emg_test=feats.scale_feats_test(emg_test,emgscaler)
            eeg_test=feats.scale_feats_test(eeg_test,eegscaler)

    emg_train_split_ML=ml.drop_ID_cols(emg_train_split_ML)
    eeg_train_split_ML=ml.drop_ID_cols(eeg_train_split_ML)
        
    sel_cols_emg=feats.sel_percent_feats_df(emg_train_split_ML,percent=15)
    sel_cols_emg=np.append(sel_cols_emg,emg_train_split_ML.columns.get_loc('Label'))
    emg_train_split_ML=emg_train_split_ML.iloc[:,sel_cols_emg]
    
    #sel_cols_eeg=feats.sel_percent_feats_df(eeg_train_split_ML,percent=3)
    sel_cols_eeg=feats.sel_feats_l1_df(eeg_train_split_ML,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
    sel_cols_eeg=np.append(sel_cols_eeg,eeg_train_split_ML.columns.get_loc('Label'))
    eeg_train_split_ML=eeg_train_split_ML.iloc[:,sel_cols_eeg]
       
    emg_model,eeg_model=train_models_opt(emg_train_split_ML,eeg_train_split_ML,args)
    
    classlabels = emg_model.classes_

    emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
    targets, predlist_emg, predlist_eeg, _, distros_emg, distros_eeg, _ = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
    
    fuser,onehotEncoder=train_lda_fuser(emg_model,eeg_model,emg_train_split_fusion,eeg_train_split_fusion,classlabels,args,sel_cols_eeg,sel_cols_emg)
    if args['stack_distros']:
        predlist_fusion=fusion.lda_fusion(fuser,onehotEncoder,distros_emg,distros_eeg,classlabels)
    else:
        predlist_fusion=fusion.lda_fusion(fuser,onehotEncoder,predlist_emg,predlist_eeg,classlabels)
    
    if args['get_train_acc']:
        emg_train=feats.scale_feats_test(emg_train,emgscaler)
        eeg_train=feats.scale_feats_test(eeg_train,eegscaler)
        emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
        traintargs, predlist_emgtrain, predlist_eegtrain, _, distros_emgtrain, distros_eegtrain, _ = refactor_synced_predict(emg_train, eeg_train, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
        if args['stack_distros']:
            predlist_train=fusion.lda_fusion(fuser,onehotEncoder,distros_emgtrain,distros_eegtrain,classlabels)
        else:
            predlist_train=fusion.lda_fusion(fuser,onehotEncoder,predlist_emgtrain,predlist_eegtrain,classlabels)
        return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train  
    else:
        return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels



def fusion_RF(emg_train, eeg_train, emg_test, eeg_test, args):
    emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_train=emg_train.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_train=eeg_train.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
            
    emg_train['ID_splitIndex']=emg_train['Label'].astype(str)+emg_train['ID_pptID'].astype(str)
    eeg_train['ID_splitIndex']=eeg_train['Label'].astype(str)+eeg_train['ID_pptID'].astype(str)
    #https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
    
    if args['trialmode']=='WithinPpt':
        sel_cols_emg=feats.sel_percent_feats_df(ml.drop_ID_cols(emg_train),percent=15)
        sel_cols_emg=np.append(sel_cols_emg,ml.drop_ID_cols(emg_train).columns.get_loc('Label'))
        sel_cols_eeg=feats.sel_feats_l1_df(ml.drop_ID_cols(eeg_train),sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
        sel_cols_eeg=np.append(sel_cols_eeg,ml.drop_ID_cols(eeg_train).columns.get_loc('Label'))
    elif args['trialmode']=='LOO':
        idx=int(eeg_test['ID_pptID'].iloc[0])-1
        sel_cols_eeg=[ml.drop_ID_cols(eeg_train).columns.get_loc(col) for col in args['eeg_feats_LOO'].iloc[idx].tolist()]
        sel_cols_emg=[ml.drop_ID_cols(emg_train).columns.get_loc(col) for col in args['emg_feats_LOO'].iloc[idx].tolist()]
    
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
        
        emg_model,eeg_model=train_models_opt(emg_train_split_ML,eeg_train_split_ML,args)    
        classlabels = emg_model.classes_
        
        targets,predlist_emg,predlist_eeg,_,distros_emg,distros_eeg,_=refactor_synced_predict(emg_train_split_fusion, eeg_train_split_fusion, emg_model, eeg_model, classlabels, args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
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
    emg_model,eeg_model=train_models_opt(emg_train,eeg_train,args)
    
    if not args['stack_distros']:
        onehotEncoder=fusion.setup_onehot(classlabels)
        onehot_pred_emg=fusion.encode_preds_onehot(fusdistros_emg,onehotEncoder)
        onehot_pred_eeg=fusion.encode_preds_onehot(fusdistros_eeg,onehotEncoder)
        fuser=fusion.train_rf_fuser(onehot_pred_emg, onehot_pred_eeg, fustargets, args['RFfuse'])
    else:
        onehotEncoder=None
        fuser=fusion.train_rf_fuser(fusdistros_emg, fusdistros_eeg, fustargets, args['RFfuse'])
    
    '---------------'
    emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
    targets, predlist_emg, predlist_eeg, _, distros_emg, distros_eeg, _  = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
    
    if args['stack_distros']:
        predlist_fusion=fusion.rf_fusion(fuser,onehotEncoder,distros_emg,distros_eeg,classlabels)
    else:
        predlist_fusion=fusion.rf_fusion(fuser,onehotEncoder,predlist_emg,predlist_eeg,classlabels)
    
    if args['get_train_acc']:
        emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
        traintargs, predlist_emgtrain, predlist_eegtrain, _, distros_emgtrain, distros_eegtrain, _ = refactor_synced_predict(emg_train, eeg_train, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
        if args['stack_distros']:
            predlist_train=fusion.rf_fusion(fuser,onehotEncoder,distros_emgtrain,distros_eegtrain,classlabels)
        else:
            predlist_train=fusion.rf_fusion(fuser,onehotEncoder,predlist_emgtrain,predlist_eegtrain,classlabels)
        return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train  
    else:
        return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels



def fusion_RF_noCV(emg_train, eeg_train, emg_test, eeg_test, args):
    emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_train=emg_train.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_train=eeg_train.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
            
    emg_train['ID_splitIndex']=emg_train['Label'].astype(str)+emg_train['ID_pptID'].astype(str)
    eeg_train['ID_splitIndex']=eeg_train['Label'].astype(str)+eeg_train['ID_pptID'].astype(str)
    #https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
    random_split=random.randint(0,100)
    emg_train_split_ML,emg_train_split_fusion=train_test_split(emg_train,test_size=0.33,random_state=random_split,stratify=emg_train[['ID_splitIndex']])
    eeg_train_split_ML,eeg_train_split_fusion=train_test_split(eeg_train,test_size=0.33,random_state=random_split,stratify=eeg_train[['ID_splitIndex']])
    #https://stackoverflow.com/questions/43095076/scikit-learn-train-test-split-can-i-ensure-same-splits-on-different-datasets
    
    
    if args['scalingtype']:
            emg_train_split_ML,emgscaler=feats.scale_feats_train(emg_train_split_ML,args['scalingtype'])
            eeg_train_split_ML,eegscaler=feats.scale_feats_train(eeg_train_split_ML,args['scalingtype'])
            emg_train_split_fusion=feats.scale_feats_test(emg_train_split_fusion,emgscaler)
            eeg_train_split_fusion=feats.scale_feats_test(eeg_train_split_fusion,eegscaler)
            emg_test=feats.scale_feats_test(emg_test,emgscaler)
            eeg_test=feats.scale_feats_test(eeg_test,eegscaler)


    emg_train_split_ML=ml.drop_ID_cols(emg_train_split_ML)
    eeg_train_split_ML=ml.drop_ID_cols(eeg_train_split_ML)
    
    
    sel_cols_emg=feats.sel_percent_feats_df(emg_train_split_ML,percent=15)
    sel_cols_emg=np.append(sel_cols_emg,emg_train_split_ML.columns.get_loc('Label'))
    emg_train_split_ML=emg_train_split_ML.iloc[:,sel_cols_emg]
    
    #sel_cols_eeg=feats.sel_percent_feats_df(eeg_train_split_ML,percent=3)
    sel_cols_eeg=feats.sel_feats_l1_df(eeg_train_split_ML,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
    sel_cols_eeg=np.append(sel_cols_eeg,eeg_train_split_ML.columns.get_loc('Label'))
    eeg_train_split_ML=eeg_train_split_ML.iloc[:,sel_cols_eeg]
       
    
    emg_model,eeg_model=train_models_opt(emg_train_split_ML,eeg_train_split_ML,args)
    
    classlabels = emg_model.classes_

    emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
    
    targets, predlist_emg, predlist_eeg, _, distros_emg, distros_eeg, _ = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
    
    fuser,onehotEncoder=train_RF_fuser(emg_model,eeg_model,emg_train_split_fusion,eeg_train_split_fusion,classlabels,args,sel_cols_eeg,sel_cols_emg)
    if args['stack_distros']:
        predlist_fusion=fusion.rf_fusion(fuser,onehotEncoder,distros_emg,distros_eeg,classlabels)
    else:
        predlist_fusion=fusion.rf_fusion(fuser,onehotEncoder,predlist_emg,predlist_eeg,classlabels)
    
    if args['get_train_acc']:
        emg_train=feats.scale_feats_test(emg_train,emgscaler)
        eeg_train=feats.scale_feats_test(eeg_train,eegscaler)
        emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
        traintargs, predlist_emgtrain, predlist_eegtrain, _, distros_emgtrain, distros_eegtrain, _ = refactor_synced_predict(emg_train, eeg_train, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
        if args['stack_distros']:
            predlist_train=fusion.rf_fusion(fuser,onehotEncoder,distros_emgtrain,distros_eegtrain,classlabels)
        else:
            predlist_train=fusion.rf_fusion(fuser,onehotEncoder,predlist_emgtrain,predlist_eegtrain,classlabels)
        return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train  
    else:
        return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels


def function_fuse_LOO(args):
    start=time.time()
    #print('emg model ',args['emg']['emg_model_type'])
    #print('eeg model ',args['eeg']['eeg_model_type'])
    if not args['data_in_memory']:
        emg_set_path=args['emg_set_path']
        eeg_set_path=args['eeg_set_path']
    
        emg_set=ml.pd.read_csv(emg_set_path,delimiter=',')
        eeg_set=ml.pd.read_csv(eeg_set_path,delimiter=',')
    else:
        emg_set=args['emg_set']
        eeg_set=args['eeg_set']
    if not args['prebalanced']: 
        emg_set,eeg_set=balance_set(emg_set,eeg_set)
    
    eeg_masks=get_ppt_split(eeg_set,args)
    emg_masks=get_ppt_split(emg_set,args)
    
    accs=[]
    emg_accs=[] #https://stackoverflow.com/questions/13520876/how-can-i-make-multiple-empty-lists-in-python
    eeg_accs=[]
    
    train_accs=[]
    
    #f1s=[]
    #emg_f1s=[]
    #eeg_f1s=[]
    '''TEMP'''
#    eeg_feat_idxs=[]
 #   eeg_feat_names=[]
    
 #   emg_feat_idxs=[]
  #  emg_feat_names=[]
 #   joint_feat_idxs=[]
  #  joint_feat_names=[]
    
    kappas=[]
    for idx,emg_mask in enumerate(emg_masks):
        eeg_mask=eeg_masks[idx]
        
        emg_ppt = emg_set[emg_mask]
        emg_others = emg_set[~emg_mask]
        eeg_ppt = eeg_set[eeg_mask]
        eeg_others = eeg_set[~eeg_mask]
        
        if args['fusion_alg']=='bayes':
            emg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            eeg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            
            index_emg=ml.pd.MultiIndex.from_arrays([emg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
            index_eeg=ml.pd.MultiIndex.from_arrays([eeg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
            emg_others=emg_others.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
            eeg_others=eeg_others.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
                    
            emg_others['ID_splitIndex']=emg_others['Label'].astype(str)+emg_others['ID_pptID'].astype(str)
            eeg_others['ID_splitIndex']=eeg_others['Label'].astype(str)+eeg_others['ID_pptID'].astype(str)
            #https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
            random_split=random.randint(0,100)
            emg_train_split_ML,emg_train_split_fusion=train_test_split(emg_others,test_size=0.33,random_state=random_split,stratify=emg_others[['ID_splitIndex']])
            eeg_train_split_ML,eeg_train_split_fusion=train_test_split(eeg_others,test_size=0.33,random_state=random_split,stratify=eeg_others[['ID_splitIndex']])
            #https://stackoverflow.com/questions/43095076/scikit-learn-train-test-split-can-i-ensure-same-splits-on-different-datasets
            
            emg_train_split_ML=ml.drop_ID_cols(emg_train_split_ML)
            eeg_train_split_ML=ml.drop_ID_cols(eeg_train_split_ML)
            
            emg_model,eeg_model=train_models_opt(emg_train_split_ML,eeg_train_split_ML,args)
            
            classlabels = emg_model.classes_
        
            emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
            targets, predlist_emg, predlist_eeg, _,_,_,_ = refactor_synced_predict(emg_ppt, eeg_ppt, emg_model, eeg_model, classlabels,args)
            
            fuser,onehotEncoder=train_bayes_fuser(emg_model,eeg_model,emg_train_split_fusion,eeg_train_split_fusion,classlabels,args)
            predlist_fusion=fusion.bayesian_fusion(fuser,onehotEncoder,predlist_emg,predlist_eeg,classlabels)
        
        elif args['fusion_alg']=='svm':
            if args['scalingtype']:
                emg_others,emgscaler=feats.scale_feats_train(emg_others,args['scalingtype'])
                eeg_others,eegscaler=feats.scale_feats_train(eeg_others,args['scalingtype'])
                emg_ppt=feats.scale_feats_test(emg_ppt,emgscaler)
                eeg_ppt=feats.scale_feats_test(eeg_ppt,eegscaler)
                
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_SVM(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
        
        elif args['fusion_alg']=='lda':
            if args['scalingtype']:
                emg_others,emgscaler=feats.scale_feats_train(emg_others,args['scalingtype'])
                eeg_others,eegscaler=feats.scale_feats_train(eeg_others,args['scalingtype'])
                emg_ppt=feats.scale_feats_test(emg_ppt,emgscaler)
                eeg_ppt=feats.scale_feats_test(eeg_ppt,eegscaler)
                
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_LDA(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
            
        elif args['fusion_alg']=='rf':
            if args['scalingtype']:
                emg_others,emgscaler=feats.scale_feats_train(emg_others,args['scalingtype'])
                eeg_others,eegscaler=feats.scale_feats_train(eeg_others,args['scalingtype'])
                emg_ppt=feats.scale_feats_test(emg_ppt,emgscaler)
                eeg_ppt=feats.scale_feats_test(eeg_ppt,eegscaler)
                
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_RF(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
       
        elif args['fusion_alg']=='hierarchical':
            
            if args['scalingtype']:
                emg_others,emgscaler=feats.scale_feats_train(emg_others,args['scalingtype'])
                eeg_others,eegscaler=feats.scale_feats_train(eeg_others,args['scalingtype'])
                emg_ppt=feats.scale_feats_test(emg_ppt,emgscaler)
                eeg_ppt=feats.scale_feats_test(eeg_ppt,eegscaler)
            
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_hierarchical(emg_others, eeg_others, emg_ppt, eeg_ppt, args)

        elif args['fusion_alg']=='hierarchical_inv':          
            
            if args['scalingtype']:
                emg_others,emgscaler=feats.scale_feats_train(emg_others,args['scalingtype'])
                eeg_others,eegscaler=feats.scale_feats_train(eeg_others,args['scalingtype'])
                emg_ppt=feats.scale_feats_test(emg_ppt,emgscaler)
                eeg_ppt=feats.scale_feats_test(eeg_ppt,eegscaler)
                                                  
            if not args['get_train_acc']:    
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_hierarchical_inv(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=fusion_hierarchical_inv(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
                 
        elif args['fusion_alg']=='featlevel':  
            
            if args['scalingtype']:
                emg_others,emgscaler=feats.scale_feats_train(emg_others,args['scalingtype'])
                eeg_others,eegscaler=feats.scale_feats_train(eeg_others,args['scalingtype'])
                emg_ppt=feats.scale_feats_test(emg_ppt,emgscaler)
                eeg_ppt=feats.scale_feats_test(eeg_ppt,eegscaler)
                            
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=feature_fusion(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
           # targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels,colsidx,colnames=feature_fusion(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
           # joint_feat_idxs.append(colsidx)
           # joint_feat_names.append(colnames)
        
        elif args['fusion_alg']=='just_emg':
            
            if args['scalingtype']:
                emg_others,emgscaler=feats.scale_feats_train(emg_others,args['scalingtype'])
                eeg_others,eegscaler=feats.scale_feats_train(eeg_others,args['scalingtype'])
                emg_ppt=feats.scale_feats_test(emg_ppt,emgscaler)
                eeg_ppt=feats.scale_feats_test(eeg_ppt,eegscaler)
            
            if not args['get_train_acc']:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=only_EMG(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=only_EMG(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
        
        elif args['fusion_alg']=='just_eeg':
            
            if args['scalingtype']:
                emg_others,emgscaler=feats.scale_feats_train(emg_others,args['scalingtype'])
                eeg_others,eegscaler=feats.scale_feats_train(eeg_others,args['scalingtype'])
                emg_ppt=feats.scale_feats_test(emg_ppt,emgscaler)
                eeg_ppt=feats.scale_feats_test(eeg_ppt,eegscaler)
            
            if not args['get_train_acc']:    
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=only_EEG(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=only_EEG(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
                    
        else:
                        
            if args['scalingtype']:
                emg_others,emgscaler=feats.scale_feats_train(emg_others,args['scalingtype'])
                eeg_others,eegscaler=feats.scale_feats_train(eeg_others,args['scalingtype'])
                emg_ppt=feats.scale_feats_test(emg_ppt,emgscaler)
                eeg_ppt=feats.scale_feats_test(eeg_ppt,eegscaler)
                
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
            
#            '''TEMP'''
 #           eeg_feat_idxs.append(sel_cols_eeg)
  #          eeg_feat_names.append(eeg_others.columns.values)
   #         emg_feat_idxs.append(sel_cols_emg)
    #        emg_feat_names.append(emg_others.columns.values)
            
            emg_model,eeg_model=train_models_opt(emg_others,eeg_others,args)
        
            classlabels = emg_model.classes_
            
            emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
                
            targets, predlist_emg, predlist_eeg, predlist_fusion,_,_,_= refactor_synced_predict(emg_ppt, eeg_ppt, emg_model, eeg_model, classlabels,args, sel_cols_eeg,sel_cols_emg)

        #acc_emg,acc_eeg,acc_fusion=evaluate_results(targets, predlist_emg, correctness_emg, predlist_eeg, correctness_eeg, predlist_fusion, correctness_fusion, classlabels)
        
        gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels=classes_from_preds(targets,predlist_emg,predlist_eeg,predlist_fusion,classlabels)
        '''could calculate log loss if got the probabilities back''' #https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
        
        #plot_confmats(gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels)
        
        if args['plot_confmats']:
            gesturelabels=[params.idx_to_gestures[label] for label in classlabels]
            tt.confmat(gest_truth,gest_pred_eeg,gesturelabels,title='EEG')
            tt.confmat(gest_truth,gest_pred_emg,gesturelabels,title='EMG')
            tt.confmat(gest_truth,gest_pred_fusion,gesturelabels,title='Fusion')
            
        emg_accs.append(accuracy_score(gest_truth,gest_pred_emg))
        eeg_accs.append(accuracy_score(gest_truth,gest_pred_eeg))
        accs.append(accuracy_score(gest_truth,gest_pred_fusion))
        
        #emg_f1s.append(f1_score(gest_truth,gest_pred_emg,average='weighted'))
        #eeg_f1s.append(f1_score(gest_truth,gest_pred_eeg,average='weighted'))
        #f1s.append(f1_score(gest_truth,gest_pred_fusion,average='weighted'))
        
        kappas.append(cohen_kappa_score(gest_truth,gest_pred_fusion))
        
        if args['get_train_acc']:
            train_truth=[params.idx_to_gestures[gest] for gest in traintargs]
            train_preds=[params.idx_to_gestures[pred] for pred in predlist_train]
            train_accs.append(accuracy_score(train_truth,train_preds))
        else:
            train_accs.append(0)
    '''TEMP'''
#    pickle.dump(eeg_feat_idxs,open(r"C:\Users\pritcham\Desktop\eeg_feat_idx.pckl",'wb'))
 #   pickle.dump(eeg_feat_names,open(r"C:\Users\pritcham\Desktop\eeg_feat_name.pckl",'wb'))
  #  pickle.dump(emg_feat_idxs,open(r"C:\Users\pritcham\Desktop\emg_feat_idx.pckl",'wb'))
   # pickle.dump(emg_feat_names,open(r"C:\Users\pritcham\Desktop\emg_feat_name.pckl",'wb'))
#    eeg_feats_idx_df=pd.DataFrame(eeg_feat_idxs)
 #   eeg_feats_idx_df.to_csv(r"C:\Users\pritcham\Desktop\eeg_feat_idx.csv",index=False,header=False)
  #  eeg_feats_df=pd.DataFrame(eeg_feat_names)
   # eeg_feats_df.to_csv(r"C:\Users\pritcham\Desktop\eeg_feat.csv",index=False,header=False)
#    emg_feats_idx_df=pd.DataFrame(emg_feat_idxs)
 #   emg_feats_idx_df.to_csv(r"C:\Users\pritcham\Desktop\emg_feat_idx.csv",index=False,header=False)
  #  emg_feats_df=pd.DataFrame(emg_feat_names)
   # emg_feats_df.to_csv(r"C:\Users\pritcham\Desktop\emg_feat.csv",index=False,header=False)
#    pickle.dump(joint_feat_idxs,open(r"C:\Users\pritcham\Desktop\joint_feat_idx.pckl",'wb'))
 #   pickle.dump(joint_feat_names,open(r"C:\Users\pritcham\Desktop\joint_feat_name.pckl",'wb'))
  #  joint_feats_idx_df=pd.DataFrame(joint_feat_idxs)
#    joint_feats_idx_df.to_csv(r"C:\Users\pritcham\Desktop\joint_feat_idx.csv",index=False,header=False)
 #   joint_feats_df=pd.DataFrame(joint_feat_names)
  #  joint_feats_df.to_csv(r"C:\Users\pritcham\Desktop\joint_feat.csv",index=False,header=False)
    
    mean_acc=stats.mean(accs)
    median_acc=stats.median(accs)
    mean_emg=stats.mean(emg_accs)
    median_emg=stats.median(emg_accs)
    mean_eeg=stats.mean(eeg_accs)
    median_eeg=stats.median(eeg_accs)
    #mean_f1_emg=stats.mean(emg_f1s)
    #mean_f1_eeg=stats.mean(eeg_f1s)
    #mean_f1_fusion=stats.mean(f1s)
    #median_f1=stats.median(f1s)
    median_kappa=stats.median(kappas)
    mean_train_acc=stats.mean(train_accs)
    end=time.time()
    #return 1-mean_acc
    return {
        'loss': 1-mean_acc,
        #'loss': 1-(median_acc*median_eeg*(1/(end-start))),
        'status': STATUS_OK,
        'median_kappa':median_kappa,
        'fusion_mean_acc':mean_acc,
        'fusion_median_acc':median_acc,
        'emg_mean_acc':mean_emg,
        'emg_median_acc':median_emg,
        'eeg_mean_acc':mean_eeg,
        'eeg_median_acc':median_eeg,
        #'emg_f1_mean':mean_f1_emg,
        #'eeg_f1_mean':mean_f1_eeg,
        #'fusion_f1_mean':mean_f1_fusion,
        'emg_accs':emg_accs,
        'eeg_accs':eeg_accs,
        'fusion_accs':accs,
        'mean_train_acc':mean_train_acc,
        'elapsed_time':end-start,}

def function_fuse_withinppt(args):
    start=time.time()
    if not args['data_in_memory']:
        emg_set_path=args['emg_set_path']
        eeg_set_path=args['eeg_set_path']
    
        emg_set=ml.pd.read_csv(emg_set_path,delimiter=',')
        eeg_set=ml.pd.read_csv(eeg_set_path,delimiter=',')
    else:
        emg_set=args['emg_set']
        eeg_set=args['eeg_set']
    if not args['prebalanced']: 
        emg_set,eeg_set=balance_set(emg_set,eeg_set)
    
    eeg_masks=get_ppt_split(eeg_set,args)
    emg_masks=get_ppt_split(emg_set,args)
    
    accs=[]
    emg_accs=[] #https://stackoverflow.com/questions/13520876/how-can-i-make-multiple-empty-lists-in-python
    eeg_accs=[]
    
    train_accs=[]
    
    #f1s=[]
    #emg_f1s=[]
    #eeg_f1s=[]
    
    kappas=[]
    for idx,emg_mask in enumerate(emg_masks):
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
        #below doesnt work, stratified by performance ie ensures theyre split
        #emg_train,emg_test=train_test_split(emg_ppt,test_size=0.33,random_state=random_split,stratify=emg_ppt[['ID_stratID']])
        #eeg_train,eeg_test=train_test_split(eeg_ppt,test_size=0.33,random_state=random_split,stratify=eeg_ppt[['ID_stratID']])
        if not emg_ppt['ID_stratID'].equals(eeg_ppt['ID_stratID']):
            raise ValueError('EMG & EEG performances misaligned')
        gest_perfs=emg_ppt['ID_stratID'].unique()
        gest_strat=pd.DataFrame([gest_perfs,[perf.split('.')[1][-1] for perf in gest_perfs]]).transpose()
        train_split,test_split=train_test_split(gest_strat,test_size=0.33,random_state=random_split,stratify=gest_strat[1])
        '''
        #below separates performances but risks unbalancing classes
        emg_train_split,emg_test_split=train_test_split(emg_ppt['ID_stratID'].unique(),test_size=0.33,random_state=random_split)
        eeg_train_split,eeg_test_split=train_test_split(eeg_ppt['ID_stratID'].unique(),test_size=0.33,random_state=random_split)

        eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(eeg_train_split)]
        eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(eeg_test_split)]
        emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(emg_train_split)]
        emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(emg_test_split)]
        '''
        eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(train_split[0])]
        eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split[0])]
        emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(train_split[0])]
        emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]
        '''REVERT BELOW, JUST FOR CHECKING IF TRAIN ACC AFFECTED BY STRATIFICATION'''
        '''       
        emg_train,emg_test=train_test_split(emg_ppt,test_size=0.33,random_state=random_split)
        eeg_train,eeg_test=train_test_split(eeg_ppt,test_size=0.33,random_state=random_split)
        '''
        
        if args['fusion_alg']=='bayes':
            emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            
            index_emg=ml.pd.MultiIndex.from_arrays([emg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
            index_eeg=ml.pd.MultiIndex.from_arrays([eeg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
            emg_train=emg_train.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
            eeg_train=eeg_train.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
                    
            emg_train['ID_splitIndex']=emg_train['Label'].astype(str)+emg_train['ID_pptID'].astype(str)
            eeg_train['ID_splitIndex']=eeg_train['Label'].astype(str)+eeg_train['ID_pptID'].astype(str)
            #https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
            random_split=random.randint(0,100)
            emg_train_split_ML,emg_train_split_fusion=train_test_split(emg_train,test_size=0.33,random_state=random_split,stratify=emg_train[['ID_splitIndex']])
            eeg_train_split_ML,eeg_train_split_fusion=train_test_split(eeg_train,test_size=0.33,random_state=random_split,stratify=eeg_train[['ID_splitIndex']])
            #https://stackoverflow.com/questions/43095076/scikit-learn-train-test-split-can-i-ensure-same-splits-on-different-datasets
            
            emg_train_split_ML=ml.drop_ID_cols(emg_train_split_ML)
            eeg_train_split_ML=ml.drop_ID_cols(eeg_train_split_ML)
            
            emg_model,eeg_model=train_models_opt(emg_train_split_ML,eeg_train_split_ML,args)
            
            classlabels = emg_model.classes_
        
            emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
            targets, predlist_emg, predlist_eeg, _,_,_,_ = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args)
            
            fuser,onehotEncoder=train_bayes_fuser(emg_model,eeg_model,emg_train_split_fusion,eeg_train_split_fusion,classlabels,args)
            predlist_fusion=fusion.bayesian_fusion(fuser,onehotEncoder,predlist_emg,predlist_eeg,classlabels)
        
        elif args['fusion_alg']=='svm':
            if args['scalingtype']:
                emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
                eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
                emg_test=feats.scale_feats_test(emg_test,emgscaler)
                eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
            
            if args['get_train_acc']:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train = fusion_SVM(emg_train, eeg_train, emg_test, eeg_test, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_SVM(emg_train, eeg_train, emg_test, eeg_test, args)
        
        elif args['fusion_alg']=='lda':
            if args['scalingtype']:
                emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
                eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
                emg_test=feats.scale_feats_test(emg_test,emgscaler)
                eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
                
            if args['get_train_acc']:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train = fusion_LDA(emg_train, eeg_train, emg_test, eeg_test, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_LDA(emg_train, eeg_train, emg_test, eeg_test, args)
        
        elif args['fusion_alg']=='rf':
            if args['scalingtype']:
                emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
                eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
                emg_test=feats.scale_feats_test(emg_test,emgscaler)
                eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
                
            if args['get_train_acc']:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train = fusion_RF(emg_train, eeg_train, emg_test, eeg_test, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_RF(emg_train, eeg_train, emg_test, eeg_test, args)
        
        elif args['fusion_alg']=='hierarchical':
                     
            if args['scalingtype']:
                emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
                eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
                emg_test=feats.scale_feats_test(emg_test,emgscaler)
                eeg_test=feats.scale_feats_test(eeg_test,eegscaler)                            
                        
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_hierarchical(emg_train, eeg_train, emg_test, eeg_test, args)

        elif args['fusion_alg']=='hierarchical_inv':
            
            if args['scalingtype']:
                emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
                eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
                emg_test=feats.scale_feats_test(emg_test,emgscaler)
                eeg_test=feats.scale_feats_test(eeg_test,eegscaler)

            if not args['get_train_acc']:            
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_hierarchical_inv(emg_train, eeg_train, emg_test, eeg_test, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train = fusion_hierarchical_inv(emg_train, eeg_train, emg_test, eeg_test, args)
                 
        elif args['fusion_alg']=='featlevel': 
            
            if args['scalingtype']:
                emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
                eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
                emg_test=feats.scale_feats_test(emg_test,emgscaler)
                eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
                            
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=feature_fusion(emg_train, eeg_train, emg_test, eeg_test, args)
        
        elif args['fusion_alg']=='just_emg':
            
            if args['scalingtype']:
                emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
                eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
                emg_test=feats.scale_feats_test(emg_test,emgscaler)
                eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
            
            if not args['get_train_acc']:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=only_EMG(emg_train, eeg_train, emg_test, eeg_test, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=only_EMG(emg_train, eeg_train, emg_test, eeg_test, args)
        
        elif args['fusion_alg']=='just_eeg':
            
            if args['scalingtype']:
                emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
                eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
                emg_test=feats.scale_feats_test(emg_test,emgscaler)
                eeg_test=feats.scale_feats_test(eeg_test,eegscaler)

            if not args['get_train_acc']:    
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=only_EEG(emg_train, eeg_train, emg_test, eeg_test, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=only_EEG(emg_train, eeg_train, emg_test, eeg_test, args)
        else:
            
            if args['scalingtype']:
                emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
                eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
                emg_test=feats.scale_feats_test(emg_test,emgscaler)
                eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
            
            if args['get_train_acc']:
                emg_trainacc=emg_train.copy()
                eeg_trainacc=eeg_train.copy()
                emg_trainacc.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
                eeg_trainacc.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
           
            
            emg_train=ml.drop_ID_cols(emg_train)
            eeg_train=ml.drop_ID_cols(eeg_train)
            
            #sel_cols_eeg=feats.sel_percent_feats_df(eeg_train,percent=3)
            sel_cols_eeg=feats.sel_feats_l1_df(eeg_train,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
            sel_cols_eeg=np.append(sel_cols_eeg,eeg_train.columns.get_loc('Label'))
            eeg_train=eeg_train.iloc[:,sel_cols_eeg]
            
            sel_cols_emg=feats.sel_percent_feats_df(emg_train,percent=15)
            sel_cols_emg=np.append(sel_cols_emg,emg_train.columns.get_loc('Label'))
            emg_train=emg_train.iloc[:,sel_cols_emg]
            
            emg_model,eeg_model=train_models_opt(emg_train,eeg_train,args)
        
            classlabels = emg_model.classes_
            
            emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
                
            targets, predlist_emg, predlist_eeg, predlist_fusion,_,_,_ = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args, sel_cols_eeg,sel_cols_emg)

            if args['get_train_acc']:
                traintargs, predlist_emgtrain, predlist_eegtrain, predlist_train,_,_,_ = refactor_synced_predict(emg_trainacc, eeg_trainacc, emg_model, eeg_model, classlabels, args, sel_cols_eeg,sel_cols_emg)

        #acc_emg,acc_eeg,acc_fusion=evaluate_results(targets, predlist_emg, correctness_emg, predlist_eeg, correctness_eeg, predlist_fusion, correctness_fusion, classlabels)
        
        gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels=classes_from_preds(targets,predlist_emg,predlist_eeg,predlist_fusion,classlabels)
        '''could calculate log loss if got the probabilities back''' #https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
        
        #plot_confmats(gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels)
        
        if args['plot_confmats']:
            gesturelabels=[params.idx_to_gestures[label] for label in classlabels]
            tt.confmat(gest_truth,gest_pred_eeg,gesturelabels,title='EEG')
            tt.confmat(gest_truth,gest_pred_emg,gesturelabels,title='EMG')
            tt.confmat(gest_truth,gest_pred_fusion,gesturelabels,title='Fusion')
            
        emg_accs.append(accuracy_score(gest_truth,gest_pred_emg))
        eeg_accs.append(accuracy_score(gest_truth,gest_pred_eeg))
        accs.append(accuracy_score(gest_truth,gest_pred_fusion))
        
        #emg_f1s.append(f1_score(gest_truth,gest_pred_emg,average='weighted'))
        #eeg_f1s.append(f1_score(gest_truth,gest_pred_eeg,average='weighted'))
        #f1s.append(f1_score(gest_truth,gest_pred_fusion,average='weighted'))
        
        kappas.append(cohen_kappa_score(gest_truth,gest_pred_fusion))
        
        if args['get_train_acc']:
            train_truth=[params.idx_to_gestures[gest] for gest in traintargs]
            train_preds=[params.idx_to_gestures[pred] for pred in predlist_train]
            train_accs.append(accuracy_score(train_truth,train_preds))
        else:
            train_accs.append(0)
        
    mean_acc=stats.mean(accs)
    median_acc=stats.median(accs)
    mean_emg=stats.mean(emg_accs)
    median_emg=stats.median(emg_accs)
    mean_eeg=stats.mean(eeg_accs)
    median_eeg=stats.median(eeg_accs)
    #mean_f1_emg=stats.mean(emg_f1s)
    #mean_f1_eeg=stats.mean(eeg_f1s)
    #mean_f1_fusion=stats.mean(f1s)
    #median_f1=stats.median(f1s)
    median_kappa=stats.median(kappas)
    mean_train_acc=stats.mean(train_accs)
    end=time.time()
    #return 1-mean_acc
    return {
        'loss': 1-mean_acc,
        'status': STATUS_OK,
        'median_kappa':median_kappa,
        'fusion_mean_acc':mean_acc,
        'fusion_median_acc':median_acc,
        'emg_mean_acc':mean_emg,
        'emg_median_acc':median_emg,
        'eeg_mean_acc':mean_eeg,
        'eeg_median_acc':median_eeg,
        #'emg_f1_mean':mean_f1_emg,
        #'eeg_f1_mean':mean_f1_eeg,
        #'fusion_f1_mean':mean_f1_fusion,
        'emg_accs':emg_accs,
        'eeg_accs':eeg_accs,
        'fusion_accs':accs,
        'mean_train_acc':mean_train_acc,
        'elapsed_time':end-start,}

def plot_opt_in_time(trials):
    fig,ax=plt.subplots()
    ax.plot(range(1, len(trials) + 1),
            [1-x['result']['loss'] for x in trials], 
        color='red', marker='.', linewidth=0)
    ax.set(title='accuracy over time')
    plt.show()
    
def plot_stat_in_time(trials,stat,ylower=0,yupper=1,showplot=True):
    fig,ax=plt.subplots()
    ax.plot(range(1, len(trials) + 1),
            [x['result'][stat] for x in trials], 
        color='red', marker='.', linewidth=0)
    #https://www.kaggle.com/code/fanvacoolt/tutorial-on-hyperopt?scriptVersionId=12981074&cellId=97
    ax.set(title=stat+' over optimisation iterations')
    ax.set_ylim(ylower,yupper)
    if showplot:
        plt.show()
    return fig
    
    # Plot something showing which were which models?
    # eg with vertical fill
    # https://stackoverflow.com/questions/23248435/fill-between-two-vertical-lines-in-matplotlib

def plot_stat_as_line(trials,stat,ylower=0,yupper=1,showplot=True):
    fig,ax=plt.subplots()
    ax.plot(range(1, len(trials) + 1),
            [x['result'][stat] for x in trials], 
        color='red')
    ax.set(title=stat+' over optimisation iterations')
    ax.set_ylim(ylower,yupper)
    if showplot:
        plt.show()
    return fig

def plot_multiple_stats(trials,stats,ylower=0,yupper=1,showplot=True):
    fig,ax=plt.subplots()
    for stat in stats:
        ax.plot(range(1, len(trials) + 1),
                [x['result'][stat] for x in trials],
                label=(stat))
    #ax.set(title=stat+' over optimisation iterations')
    ax.legend()#loc='upper center')
    ax.set_ylim(ylower,yupper)
    if showplot:
        plt.show()
    return fig

def calc_runningbest(trials,stat=None):
    if stat is None:
        best=np.maximum.accumulate([1-x['result']['loss'] for x in trials])
    else:
        best=np.maximum.accumulate([x['result'][stat] for x in trials])
    return best

def plot_multiple_stats_with_best(trials,stats,runbest=None,ylower=0,yupper=1,showplot=True):
    if isinstance(trials,pd.DataFrame):
        fig=plot_multi_runbest_df(trials,stats,runbest,ylower,yupper,showplot)
        return fig
    fig,ax=plt.subplots()
    for stat in stats:
        ax.plot(range(1, len(trials) + 1),
                [x['result'][stat] for x in trials],
                label=(stat))
        #https://www.kaggle.com/code/fanvacoolt/tutorial-on-hyperopt?scriptVersionId=12981074&cellId=97
    #ax.set(title=stat+' over optimisation iterations')
    best=calc_runningbest(trials,runbest)
    ax.plot(range(1,len(trials)+1),best,label='running best')
    ax.legend()#loc='upper center')
    ax.set_ylim(ylower,yupper)
    if showplot:
        plt.show()
    return fig

def plot_multi_runbest_df(trials,stats,runbest,ylower,yupper,showplot):
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
  
def scatterbox(trials,stat='fusion_accs',ylower=0,yupper=1,showplot=True):
    fig,ax=plt.subplots()
    X=range(1, len(trials) + 1)
    H=[x['result'][stat] for x in trials]
    groups = [[] for i in range(max(X))]
    [groups[X[i]-1].append(H[i]) for i in range(len(H))]
    groups=[each[0] for each in groups]
    ax.boxplot(groups,showmeans=True)
    ax.set(title=stat+' over optimisation iterations')
    ax.set_ylim(ylower,yupper)
    if showplot:
        plt.show()
    #https://stackoverflow.com/questions/53473733/how-to-add-box-plot-to-scatter-data-in-matplotlib
    return fig
        
def setup_search_space(architecture,include_svm):
    emgoptions=[
                {'emg_model_type':'RF',
                 'n_trees':scope.int(hp.quniform('emg.RF.ntrees',10,100,q=5)),
                 'max_depth':5,#scope.int(hp.quniform('emg.RF.maxdepth',2,5,q=1)),
                 #integerising search space https://github.com/hyperopt/hyperopt/issues/566#issuecomment-549510376
                 },
                {'emg_model_type':'kNN',
                 'knn_k':scope.int(hp.quniform('emg.knn.k',1,25,q=1)),
                 },
                {'emg_model_type':'LDA',
                 'LDA_solver':hp.choice('emg.LDA_solver',['svd','lsqr','eigen']), #removed lsqr due to LinAlgError: SVD did not converge in linear least squares but readding as this has not repeated
                 'shrinkage':hp.uniform('emg.lda.shrinkage',0.0,1.0),
                 },
                {'emg_model_type':'QDA', #emg qda reg 0.3979267 actually worked well!! for withinppt
                 'regularisation':hp.uniform('emg.qda.regularisation',0.0,1.0), #https://www.kaggle.com/code/code1110/best-parameter-s-for-qda/notebook
                 },
                {'emg_model_type':'gaussNB',
                 'smoothing':hp.loguniform('emg.gnb.smoothing',np.log(1e-9),np.log(1e0)),
                 },
 #               {'emg_model_type':'SVM',    #SKL SVC likely unviable, excessively slow
  #               'svm_C':hp.uniform('emg.svm.c',0.1,100), #use loguniform?
   #              },
                ]
    eegoptions=[
                {'eeg_model_type':'RF',
                 'n_trees':scope.int(hp.quniform('eeg_ntrees',10,100,q=5)),
                 'max_depth':5,#scope.int(hp.quniform('eeg.RF.maxdepth',2,5,q=1)),
                 },
                {'eeg_model_type':'kNN', #Previously discounted EEG KNN due to reliably slow & low results
                 'knn_k':scope.int(hp.quniform('eeg.knn.k',1,25,q=1)),
                 },
                {'eeg_model_type':'LDA',
                 'LDA_solver':hp.choice('eeg.LDA_solver',['svd','lsqr','eigen']),
                 'shrinkage':hp.uniform('eeg.lda.shrinkage',0.0,1.0),
                 },
                {'eeg_model_type':'QDA',
                 'regularisation':hp.uniform('eeg.qda.regularisation',0.0,1.0),
                 },
                {'eeg_model_type':'gaussNB',
                 'smoothing':hp.loguniform('eeg.gnb.smoothing',np.log(1e-9),np.log(1e0)),
                 },
 #               {'eeg_model_type':'SVM',    #SKL SVC likely unviable, excessively slow
  #               'svm_C':hp.uniform('eeg.svm.c',0.1,100), #use loguniform?
   #              },
                ]
    if include_svm:
        emgoptions.append({'emg_model_type':'SVM_PlattScale',
                 'kernel':hp.choice('emg.svm.kernel',['rbf']),#'poly','linear']),
                 'svm_C':hp.loguniform('emg.svm.c',np.log(0.1),np.log(100)), #use loguniform? #https://queirozf.com/entries/choosing-c-hyperparameter-for-svm-classifiers-examples-with-scikit-learn
                 'gamma':hp.loguniform('emg.svm.gamma',np.log(0.01),np.log(1)), #maybe log, from lower? #https://vitalflux.com/svm-rbf-kernel-parameters-code-sample/
                 #eg sklearns gridsearch doc uses SVC as an example with C log(1e0,1e3) & gamma log(1e-4,1e-3)
                 })
        eegoptions.append({'eeg_model_type':'SVM_PlattScale',
                 'kernel':hp.choice('eeg.svm.kernel',['rbf']),#'poly','linear']),
                 'svm_C':hp.loguniform('eeg.svm.c',np.log(0.01),np.log(100)),
                 'gamma':hp.loguniform('eeg.svm.gamma',np.log(0.01),np.log(1)), 
                 #https://www.kaggle.com/code/donkeys/exploring-hyperopt-parameter-tuning?scriptVersionId=12655875&cellId=64
                 # naming convention https://github.com/hyperopt/hyperopt/issues/380#issuecomment-685173200
                 })
    space = {
            'emg':hp.choice('emg model',emgoptions),
            'eeg':hp.choice('eeg model',eegoptions),
            'svmfuse':{
                #'kernel':hp.choice('eeg.svm.kernel',['rbf']),#'poly','linear']),
                'svm_C':hp.loguniform('fus.svm.c',np.log(0.01),np.log(100)),
                #'gamma':hp.loguniform('eeg.svm.gamma',np.log(0.01),np.log(100)),
                },
            'ldafuse':{
                'LDA_solver':hp.choice('fus.LDA.solver',['svd','lsqr','eigen']),
                'shrinkage':hp.uniform('fus.lda.shrinkage',0.0,1.0),
                },
            'RFfuse':{
                'n_trees':scope.int(hp.quniform('fus.RF.ntrees',10,100,q=5)),
                'max_depth':5,#scope.int(hp.quniform('fus.RF.maxdepth',2,5,q=1)),
                },
            'eeg_weight_opt':hp.uniform('fus.optEEG.EEGweight',0.0,100.0),
            'fusion_alg':hp.choice('fusion algorithm',[
                'mean',
                '3_1_emg',
                '3_1_eeg',
                'opt_weight',
                #'bayes', # NEED TO IMPLEMENT SCALING AND SELECTION
                'highest_conf',
                'svm',
                'lda',
                'rf',
                ]),
            #'emg_set_path':params.emg_set_path_for_system_tests,
            #'eeg_set_path':params.eeg_set_path_for_system_tests,
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
    
    if architecture=='featlevel':
        if include_svm:
            space.update({
                'fusion_alg':hp.choice('fusion algorithm',['featlevel',]),
                'featfuse_sel_feats_together':False,#hp.choice('selfeatstogether',[True,False]),
                #somehow when this is true theres an issue with the Label col. think we end up with
                #either two label cols or none depending on whether we add it to selcols_emgeeg twice
                'featfuse':hp.choice('featfuse model',[
                    {'featfuse_model_type':'RF',
                     'n_trees':scope.int(hp.quniform('featfuse.RF.ntrees',10,100,q=5)),
                     'max_depth':5,#scope.int(hp.quniform('featfuse.RF.maxdepth',2,5,q=1)),
                     },
                    {'featfuse_model_type':'kNN',
                     'knn_k':scope.int(hp.quniform('featfuse.knn.k',1,25,q=1)),
                     },
                    {'featfuse_model_type':'LDA',
                     'LDA_solver':hp.choice('featfuse.LDA_solver',['svd','lsqr','eigen']),
                     'shrinkage':hp.uniform('featfuse.lda.shrinkage',0.0,1.0),
                     },
                    {'featfuse_model_type':'QDA',
                     'regularisation':hp.uniform('featfuse.qda.regularisation',0.0,1.0), #https://www.kaggle.com/code/code1110/best-parameter-s-for-qda/notebook
                     },
                    {'featfuse_model_type':'gaussNB',
                     'smoothing':hp.loguniform('featfuse.gnb.smoothing',np.log(1e-9),np.log(1e0)),
                     },
                    {'featfuse_model_type':'SVM_PlattScale', #keep this commented out
                     'kernel':hp.choice('featfuse.svm.kernel',['rbf']),#'poly','linear']),
                     'svm_C':hp.loguniform('featfuse.svm.c',np.log(0.01),np.log(100)),
                     'gamma':hp.loguniform('featfuse.svm.gamma',np.log(0.01),np.log(1)),
                     },
                    ]),
                })
        else:
            space.update({
                'fusion_alg':hp.choice('fusion algorithm',['featlevel',]),
                'featfuse_sel_feats_together':False,#hp.choice('selfeatstogether',[True,False]),
                #somehow when this is true theres an issue with the Label col. think we end up with
                #either two label cols or none depending on whether we add it to selcols_emgeeg twice
                'featfuse':hp.choice('featfuse model',[
                    {'featfuse_model_type':'RF',
                     'n_trees':scope.int(hp.quniform('featfuse.RF.ntrees',10,100,q=5)),
                     'max_depth':5,#scope.int(hp.quniform('featfuse.RF.maxdepth',2,5,q=1)),
                     },
                    {'featfuse_model_type':'kNN',
                     'knn_k':scope.int(hp.quniform('featfuse.knn.k',1,25,q=1)),
                     },
                    {'featfuse_model_type':'LDA',
                     'LDA_solver':hp.choice('featfuse.LDA_solver',['svd','lsqr','eigen']),
                     'shrinkage':hp.uniform('featfuse.lda.shrinkage',0.0,1.0),
                     },
                    {'featfuse_model_type':'QDA',
                     'regularisation':hp.uniform('featfuse.qda.regularisation',0.0,1.0), #https://www.kaggle.com/code/code1110/best-parameter-s-for-qda/notebook
                     },
                    {'featfuse_model_type':'gaussNB',
                     'smoothing':hp.loguniform('featfuse.gnb.smoothing',np.log(1e-9),np.log(1e0)),
                     },
                    ]),
                })
        space.pop('emg',None)
        space.pop('eeg',None)
        space.pop('svmfuse',None)
        space.pop('ldafuse',None)
        space.pop('RFfuse',None)
        space.pop('eeg_weight_opt',None)
        
    elif architecture=='hierarchical':
        space.update({'fusion_alg':hp.choice('fusion algorithm',['hierarchical',])})
        space.pop('svmfuse',None)
        space.pop('ldafuse',None)
        space.pop('RFfuse',None)
        space.pop('eeg_weight_opt',None)
        
    elif architecture=='hierarchical_inv':
        space.update({'fusion_alg':hp.choice('fusion algorithm',['hierarchical_inv',])})
        space.pop('svmfuse',None)
        space.pop('ldafuse',None)
        space.pop('RFfuse',None)
        space.pop('eeg_weight_opt',None)
        
    elif architecture=='just_emg':
        space.update({'fusion_alg':hp.choice('fusion algorithm',['just_emg',])})
        space.pop('eeg',None)
        space.pop('svmfuse',None)
        space.pop('ldafuse',None)
        space.pop('RFfuse',None)
        space.pop('eeg_weight_opt',None)
    
    elif architecture=='just_eeg':
        space.update({'fusion_alg':hp.choice('fusion algorithm',['just_eeg',])})
        space.pop('emg',None)
        space.pop('svmfuse',None)
        space.pop('ldafuse',None)
        space.pop('RFfuse',None)
        space.pop('eeg_weight_opt',None)
        
    return space


def optimise_fusion(trialmode,prebalance=True,architecture='decision',platform='not server',iters=35):
    incl_svm = True if trialmode=='WithinPpt' else False
    space=setup_search_space(architecture,incl_svm)
    space.update({'trialmode':trialmode})
    
    if platform=='server':
        space.update({'emg_set_path':params.jeong_EMGfeats_server,
                      #'eeg_set_path':params.jeong_EEGfeats_server})
                      #'eeg_set_path':params.jeong_RawEEGfeats_server})
                      'eeg_set_path':params.jeong_noCSP_WidebandFeats_server})
    
    if prebalance:
        emg_set=ml.pd.read_csv(space['emg_set_path'],delimiter=',')
        eeg_set=ml.pd.read_csv(space['eeg_set_path'],delimiter=',')
        emg_set,eeg_set=balance_set(emg_set,eeg_set)
        space.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True})
        
    trials=Trials() #http://hyperopt.github.io/hyperopt/getting-started/minimizing_functions/#attaching-extra-information-via-the-trials-object
    
    if trialmode=='LOO':
        space.update({'l1_sparsity':0.005}) #0.00015
        #space.update({'l1_maxfeats':240}) #this would be sqrt(57600) ie size of train set.
        space.update({'l1_maxfeats':88}) #88 consistent with emg, LOO didnt overfit so not reducing further
        '''DONT necessarily need to do for generalist as not overfitting, switch out in algo funcs'''
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        emgeegcols=pd.read_csv(params.jointemgeegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,
                      'jointemgeeg_feats_LOO':emgeegcols,})
        best = fmin(function_fuse_LOO,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=iters,
                    trials=trials)
        
    elif trialmode=='WithinPpt':
        space.update({'l1_sparsity':0.005}) #0.002
        space.update({'l1_maxfeats':40}) # sqrt(2400*0.66)=sqrt(1584), ie size of train set
        best = fmin(function_fuse_withinppt,
                space=space,
                algo=tpe.suggest,
                max_evals=iters,
                trials=trials)
    else:
        raise ValueError('Unrecognised testing strategy, should be LOO or WithinPpt')
        
    return best, space, trials
    

def save_resultdict(filepath,resultdict,dp=4):
    '''also get the input arguments and print those?'''
    
    #https://stackoverflow.com/questions/61894745/write-dictionary-to-text-file-with-newline
    #sig fig would be nicer https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    status=resultdict['Results'].pop('status')
    emg_accs=resultdict['Results'].pop('emg_accs',None)
    eeg_accs=resultdict['Results'].pop('eeg_accs',None)
    fusion_accs=resultdict['Results'].pop('fusion_accs',None)
    
    f=open(filepath,'w')
    try:
        target=list(resultdict['Results'].keys())[list(resultdict['Results'].values()).index(1-resultdict['Results']['loss'])]
        f.write(f"Optimising for {target}\n\n")
    except ValueError:
        target, _ = min(resultdict['Results'].items(), key=lambda x: abs(1-resultdict['Results']['loss'] - x[1]))
        f.write(f"Probably optimising for {target}\n\n")
    
    if 'eeg' in resultdict['Chosen parameters']:
        f.write('EEG Parameters:\n')
        for k in resultdict['Chosen parameters']['eeg'].keys():
            f.write(f"\t'{k}':'{round(resultdict['Chosen parameters']['eeg'][k],dp)if not isinstance(resultdict['Chosen parameters']['eeg'][k],str) else resultdict['Chosen parameters']['eeg'][k]}'\n")
    
    if 'emg' in resultdict['Chosen parameters']:
        f.write('EMG Parameters:\n')
        for k in resultdict['Chosen parameters']['emg'].keys():
            f.write(f"\t'{k}':'{round(resultdict['Chosen parameters']['emg'][k],dp)if not isinstance(resultdict['Chosen parameters']['emg'][k],str) else resultdict['Chosen parameters']['emg'][k]}'\n")
    
    f.write('Fusion algorithm:\n')
    f.write(f"\t'{'fusion_alg'}':'{resultdict['Chosen parameters']['fusion_alg']}'\n")
    if resultdict['Chosen parameters']['fusion_alg']=='featlevel':
        f.write('Feature-level Fusion Parameters:\n')
        for k in resultdict['Chosen parameters']['featfuse'].keys():
            f.write(f"\t'{k}':'{round(resultdict['Chosen parameters']['featfuse'][k],dp)if not isinstance(resultdict['Chosen parameters']['featfuse'][k],str) else resultdict['Chosen parameters']['featfuse'][k]}'\n")
    
    f.write('Results:\n')
    resultdict['Results']['status']=status
    for k in resultdict['Results'].keys():
        f.write(f"\t'{k}':'{round(resultdict['Results'][k],dp)if not isinstance(resultdict['Results'][k],str) else resultdict['Results'][k]}'\n")
    
    resultdict['Results']['emg_accs']=emg_accs
    resultdict['Results']['eeg_accs']=eeg_accs
    resultdict['Results']['fusion_accs']=fusion_accs
    f.close()

def load_results_obj(path):
    load_trials=pickle.load(open(path,'rb'))
    load_table=pd.DataFrame(load_trials.trials)
    load_table_readable=pd.concat(
        [pd.DataFrame(load_table['result'].tolist()),
         pd.DataFrame(pd.DataFrame(load_table['misc'].tolist())['vals'].values.tolist())],
        axis=1,join='outer')
    return load_trials,load_table,load_table_readable

if __name__ == '__main__':
    
    if len(sys.argv)>1:
        architecture=sys.argv[1]
        trialmode=sys.argv[2]
        platform=sys.argv[3]
        if platform=='not_server':
            platform='not server'
        if len(sys.argv)>4:
            num_iters=int(sys.argv[4])
        if len(sys.argv)>5:
            showplots=sys.argv[5].lower()
        else:
            showplots=None
    else:
        architecture='decision'    
        trialmode='WithinPpt'
        platform='not server'
        num_iters=1
        showplots=None
        
    if architecture not in ['decision','featlevel','hierarchical','hierarchical_inv','just_emg','just_eeg']:
        errstring=('requested architecture '+architecture+' not recognised, expecting one of:\n decision\n featlevel\n hierarchical\n hierarchical_inv')
        raise KeyboardInterrupt(errstring)
        
    if (platform=='server') or (showplots=='false'):
        showplot_toggle=False
    else:
        showplot_toggle=True


    best,space,trials=optimise_fusion(trialmode=trialmode,architecture=architecture,platform=platform,iters=num_iters)
    
    #space=stochastic.sample(setup_search_space())
    #best_results=function_fuse_LOO(space)
    #raise
        
    if 1:  #if showplots ?  
        chosen_space=space_eval(space,best)
        chosen_space['plot_confmats']=True
        if trialmode=='LOO':
            chosen_results=function_fuse_LOO(chosen_space)
        elif trialmode=='WithinPpt':
            chosen_results=function_fuse_withinppt(chosen_space)
    '''
    start_prebal=time.time()
    best,space,trials=optimise_fusion()
    t_prebal=time.time()-start_prebal
    
    start_manbal=time.time()
    best,space,trials=optimise_fusion(prebalance=False)
    t_manbal=time.time()-start_manbal
    print(f"Time holding balanced set in memory: {t_prebal}\nTime reading set every time: {t_manbal}")
    '''
    #best,space,trials=optimise_fusion()
    
    if 0:
        '''performing a whole fresh evaluation with the chosen params'''
        best_results=function_fuse_LOO(space_eval(space,best))
    else:
        best_results=trials.best_trial['result']
        #https://stackoverflow.com/questions/20776502/where-to-find-the-loss-corresponding-to-the-best-configuration-with-hyperopt
    #could just get trials.results?
    
    bestparams=space_eval(space,best)
    
    for static in ['eeg_set_path','emg_set_path','using_literature_data']:
        bestparams.pop(static)
    bestparams.pop('eeg_set')
    bestparams.pop('emg_set')
    bestparams.pop('eeg_feats_LOO',None)
    #bestparams.drop('emg_feats_LOO',errors='ignore') #if a DF   
    bestparams.pop('emg_feats_LOO',None)
    bestparams.pop('jointemgeeg_feats_LOO',None)
    
    print(bestparams)
   # print('Best Coehns Kappa between ground truth and fusion predictions: ',
    #      1-(best_results['loss']))
    print('Best mean Fusion accuracy: ',1-best_results['loss'])
         
    winner={'Chosen parameters':bestparams,
            'Results':best_results}
    
    table=pd.DataFrame(trials.trials)
    table_readable=pd.concat(
        [pd.DataFrame(table['result'].tolist()),
         pd.DataFrame(pd.DataFrame(table['misc'].tolist())['vals'].values.tolist())],
        axis=1,join='outer')
    
    '''SETTING RESULT PATH'''
    currentpath=os.path.dirname(__file__)
    result_dir=params.jeong_results_dir
    resultpath=os.path.join(currentpath,result_dir)    
    resultpath=os.path.join(resultpath,'Fusion_CSP',trialmode)
    
    '''PICKLING THE TRIALS OBJECT'''
    trials_obj_path=os.path.join(resultpath,'trials_obj.p')
    pickle.dump(trials,open(trials_obj_path,'wb'))
    '''CODE FOR LOADING TRIALS OBJ'''
    #load_trials_var=pickle.load(open(filename,'rb'))
    #load_trials,load_table,load_table_readable=load_results_obj(filepath)
    
    '''saving best parameters & results'''
    reportpath=os.path.join(resultpath,'params_results_report.txt')
    save_resultdict(reportpath,winner)
    
    
    if architecture=='featlevel':
        fus_acc_plot=plot_stat_in_time(trials,'fusion_mean_acc',showplot=showplot_toggle)
        acc_compare_plot=plot_multiple_stats_with_best(trials,['fusion_mean_acc'],runbest='fusion_mean_acc',showplot=showplot_toggle)  
        fus_acc_box=scatterbox(trials,'fusion_accs',showplot=showplot_toggle)
        
        '''saving figures of performance over time'''
        fus_acc_plot.savefig(os.path.join(resultpath,'fusion_acc.png'))
        acc_compare_plot.savefig(os.path.join(resultpath,'acc_compare.png'))
        fus_acc_box.savefig(os.path.join(resultpath,'fusion_box.png'))
        
        per_fusalg=boxplot_param(table_readable,'featfuse model','fusion_mean_acc',showplot=showplot_toggle)
        per_fusalg.savefig(os.path.join(resultpath,'fus_alg.png'))
        
    elif architecture=='just_emg':
        emg_acc_plot=plot_stat_in_time(trials,'emg_mean_acc',showplot=showplot_toggle)
        acc_compare_plot=plot_multiple_stats_with_best(trials,['emg_mean_acc'],runbest='emg_mean_acc',showplot=showplot_toggle)  
        # BELOW IF REPORTING TRAIN ACCURACY
        #acc_compare_plot=plot_multiple_stats_with_best(trials,['emg_mean_acc','mean_train_acc'],runbest='emg_mean_acc',showplot=showplot_toggle)
        emg_acc_box=scatterbox(trials,'emg_accs',showplot=showplot_toggle)
        
        '''saving figures of performance over time'''
        emg_acc_plot.savefig(os.path.join(resultpath,'emgOnly_acc.png'))
        acc_compare_plot.savefig(os.path.join(resultpath,'emgOnly_acc_compare.png'))
        emg_acc_box.savefig(os.path.join(resultpath,'emgOnly_box.png'))
        
        per_emgmodel=boxplot_param(table_readable,'emg model','emg_mean_acc',showplot=showplot_toggle)
        per_emgmodel.savefig(os.path.join(resultpath,'emgOnly_model.png'))
        
    elif architecture=='just_eeg':
        eeg_acc_plot=plot_stat_in_time(trials,'eeg_mean_acc',showplot=showplot_toggle)
        acc_compare_plot=plot_multiple_stats_with_best(trials,['eeg_mean_acc'],runbest='eeg_mean_acc',showplot=showplot_toggle)  
        # BELOW IF REPORTING TRAIN ACCURACY
        #acc_compare_plot=plot_multiple_stats_with_best(trials,['eeg_mean_acc','mean_train_acc'],runbest='eeg_mean_acc',showplot=showplot_toggle)
        eeg_acc_box=scatterbox(trials,'eeg_accs',showplot=showplot_toggle)
        
        '''saving figures of performance over time'''
        eeg_acc_plot.savefig(os.path.join(resultpath,'eegOnly_acc.png'))
        acc_compare_plot.savefig(os.path.join(resultpath,'eegOnly_acc_compare.png'))
        eeg_acc_box.savefig(os.path.join(resultpath,'eegOnly_box.png'))
        
        per_eegmodel=boxplot_param(table_readable,'eeg model','eeg_mean_acc',showplot=showplot_toggle)
        per_eegmodel.savefig(os.path.join(resultpath,'eegOnly_model.png'))
    
    else:
    
        emg_acc_plot=plot_stat_in_time(trials, 'emg_mean_acc',showplot=showplot_toggle)
        eeg_acc_plot=plot_stat_in_time(trials, 'eeg_mean_acc',showplot=showplot_toggle)
        #plot_stat_in_time(trials, 'loss')
        fus_acc_plot=plot_stat_in_time(trials,'fusion_mean_acc',showplot=showplot_toggle)
        #plot_stat_in_time(trials,'elapsed_time',0,200)
        
        acc_compare_plot=plot_multiple_stats_with_best(trials,['emg_mean_acc','eeg_mean_acc','fusion_mean_acc'],runbest='fusion_mean_acc',showplot=showplot_toggle)
        # BELOW IF REPORTING TRAIN ACCURACY
        #acc_compare_plot=plot_multiple_stats_with_best(trials,['emg_mean_acc','eeg_mean_acc','fusion_mean_acc','mean_train_acc'],runbest='fusion_mean_acc',showplot=showplot_toggle)

        emg_acc_box=scatterbox(trials,'emg_accs',showplot=showplot_toggle)
        eeg_acc_box=scatterbox(trials,'eeg_accs',showplot=showplot_toggle)
        fus_acc_box=scatterbox(trials,'fusion_accs',showplot=showplot_toggle)
        
        #print('plotting ppt1 just to get a confmat')
        #ppt1acc=function_fuse_pptn(space_eval(space,best),1,plot_confmats=True)
    
        '''saving figures of performance over time'''
        emg_acc_plot.savefig(os.path.join(resultpath,'emg_acc.png'))
        eeg_acc_plot.savefig(os.path.join(resultpath,'eeg_acc.png'))
        fus_acc_plot.savefig(os.path.join(resultpath,'fusion_acc.png'))
        acc_compare_plot.savefig(os.path.join(resultpath,'acc_compare.png'))
        
        emg_acc_box.savefig(os.path.join(resultpath,'emg_box.png'))
        eeg_acc_box.savefig(os.path.join(resultpath,'eeg_box.png'))
        fus_acc_box.savefig(os.path.join(resultpath,'fusion_box.png'))
        
    #for properly evaluating results later: https://towardsdatascience.com/multiclass-classification-evaluation-with-roc-curves-and-roc-auc-294fd4617e3a
        '''figures of performance per model choice'''
        per_emgmodel=boxplot_param(table_readable,'emg model','fusion_mean_acc',showplot=showplot_toggle)
        per_eegmodel=boxplot_param(table_readable,'eeg model','fusion_mean_acc',showplot=showplot_toggle)
        per_fusalg=boxplot_param(table_readable,'fusion algorithm','fusion_mean_acc',showplot=showplot_toggle)
        
        per_emgmodel.savefig(os.path.join(resultpath,'emg_model.png'))
        per_eegmodel.savefig(os.path.join(resultpath,'eeg_model.png'))
        per_fusalg.savefig(os.path.join(resultpath,'fus_alg.png'))
        
        #boxplot_param(table_readable,'emg model','emg_mean_acc')
        #boxplot_param(table_readable,'eeg model','eeg_mean_acc')
         
        '''
        table_readable['delta']=table_readable['fusion_mean_acc']-table_readable['emg_mean_acc']
table_readable['delta exclude']=np.where(table_readable['emg_mean_acc'] >0.49, table_readable['delta'],0)
table_readable['delta exclude 2']=np.where(table_readable['fusion_mean_acc'] >0.49, table_readable['delta'],0)
boxplot_param(table_readable,'eeg model','delta exclude 2',ylower=-0.3,yupper=0.3)
'''
#table_readable['delta exclude 3']=np.where(table_readable['emg_mean_acc'] >0.6, table_readable['delta'],0)
#table_readable['delta exclude 4']=np.where(table_readable['emg_mean_acc'] >table_readable['eeg_mean_acc'], table_readable['delta'],0)
#boxplot_param(table_readable,'fusion algorithm','delta exclude',ylower=-0.3,yupper=0.3)

#eegtable_readable['overfit level']=eegtable_readable['mean_train_acc']-eegtable_readable['eeg_mean_acc']
#table_readable['overfit level']=np.where(table_readable['delta'] <0, abs(table_readable['delta']),None)
#table_readable['train_acc_nonzero']=np.where(table_readable['mean_train_acc']>0,table_readable['mean_train_acc'],np.nan)
#table_readable['acc_above_chance']=np.where(table_readable['emg_mean_acc']>0.26,table_readable['emg_mean_acc'],np.nan)
        
        '''
        acc_per_emg_model=table_readable.sort_values('emg model').plot(x='emg model',y='fusion_mean_acc',style='o')
        acc_per_eeg_model=table_readable.sort_values('eeg model').plot(x='eeg model',y='fusion_mean_acc',style='o')
        acc_per_fus_alg=table_readable.sort_values('fusion algorithm').plot(x='fusion algorithm',y='fusion_mean_acc',style='o')
        
        acc_per_emg_model.figure.savefig(os.path.join(resultpath,'emg_model.png'))
        acc_per_eeg_model.figure.savefig(os.path.join(resultpath,'eeg_model.png'))
        acc_per_fus_alg.figure.savefig(os.path.join(resultpath,'fus_alg.png'))
        '''
    
    raise KeyboardInterrupt('ending execution here!')
                                
    
    '''per_ppt_accs = ml.pd.DataFrame(list(zip(pptIDs,emg_accs,eeg_accs,fus_accs)),columns=['pptID','emg_acc','eeg_acc','fusion_acc'])'''
    