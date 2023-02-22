#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 23:42:00 2022

@author: pritcham
"""

import os
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
from sklearn.model_selection import train_test_split
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

def balance_set(emg_set,eeg_set):
    #print('initial')
    #_,_=inspect_set_balance(emg_set=emg_set,eeg_set=eeg_set)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_set[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_set[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg=emg_set.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg=eeg_set.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    
    emg.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    eeg['ID_stratID']=eeg['ID_pptID'].astype(str)+eeg['Label'].astype(str)
    emg['ID_stratID']=emg['ID_pptID'].astype(str)+emg['Label'].astype(str)
    
    stratsize=np.min(emg['ID_stratID'].value_counts())
    balemg = emg.groupby('ID_stratID')
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
        
        #distro_fusion=fusion.fuse_mean(distro_emg,distro_eeg)
        distro_fusion=fusion.fuse_select(distro_emg, distro_eeg, args)
        pred_fusion=ml.pred_from_distro(classlabels,distro_fusion)
       # distrolist_fusion.append(distro_fusion)
        predlist_fusion.append(pred_fusion)
            
        targets.append(TargetLabel)
    return targets, predlist_emg, predlist_eeg, predlist_fusion

def refactor_synced_predict(test_set_emg,test_set_eeg,model_emg,model_eeg,classlabels,args):
  #  distrolist_emg=[]
    predlist_emg=[]
    
 #   distrolist_eeg=[]
    predlist_eeg=[]
    
   # distrolist_fusion=[]
    predlist_fusion=[]
    
    targets=[]
    
    index_emg=ml.pd.MultiIndex.from_arrays([test_set_emg[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([test_set_eeg[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg=test_set_emg.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg=test_set_eeg.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    for index,emgrow in emg.iterrows():
        eegrow = eeg[(eeg['ID_pptID']==emgrow['ID_pptID'])
                              & (eeg['ID_run']==emgrow['ID_run'])
                              & (eeg['Label']==emgrow['Label'])
                              & (eeg['ID_gestrep']==emgrow['ID_gestrep'])
                              & (eeg['ID_tend']==emgrow['ID_tend'])]
        #syntax like the below would do it closer to a .where
        #eegrow=test_set_eeg[test_set_eeg[['ID_pptID','Label']]==emgrow[['ID_pptID','Label']]]
        if eegrow.empty:
            print('No matching EEG window for EMG window '+str(emgrow['ID_pptID'])+str(emgrow['ID_run'])+str(emgrow['Label'])+str(emgrow['ID_gestrep'])+str(emgrow['ID_tend']))
            continue
        
        TargetLabel=emgrow['Label']
        if TargetLabel != eegrow['Label'].values:
            raise Exception('Sense check failed, target label should agree between modes')
        targets.append(TargetLabel)
        
    '''Get values from instances'''
    IDs=list(emg.filter(regex='^ID_').keys())
    IDs.append('Label')
    emgvals=emg.drop(IDs,axis='columns').values
    eegvals=eeg.drop(IDs,axis='columns').values
    
    '''Pass values to models'''
    
    distros_emg=ml.prob_dist(model_emg,emgvals)
    for distro in distros_emg:
        pred_emg=ml.pred_from_distro(classlabels,distro)
   # distrolist_emg.append(distro_emg)
        predlist_emg.append(pred_emg)
    
    distros_eeg=ml.prob_dist(model_eeg,eegvals)
    for distro in distros_eeg:
        pred_eeg=ml.pred_from_distro(classlabels,distro)
   # distrolist_eeg.append(distro_eeg)
        predlist_eeg.append(pred_eeg)
    
    #distro_fusion=fusion.fuse_mean(distro_emg,distro_eeg)
    distros_fusion=fusion.fuse_select(distros_emg, distros_eeg, args)
    for distro in distros_fusion:
        pred_fusion=ml.pred_from_distro(classlabels,distro)
       # distrolist_fusion.append(distro_fusion)
        predlist_fusion.append(pred_fusion) 
        
    return targets, predlist_emg, predlist_eeg, predlist_fusion

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
    emg_model = ml.train_optimise(emg_train_set,emg_model_type,args['emg'])
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
    targets,predlist_emg,predlist_eeg,_=refactor_synced_predict(emg_set, eeg_set, model_emg, model_eeg, classlabels, args)
    onehot=fusion.setup_onehot(classlabels)
    onehot_pred_emg=fusion.encode_preds_onehot(predlist_emg,onehot)
    onehot_pred_eeg=fusion.encode_preds_onehot(predlist_eeg,onehot)
    fuser=fusion.train_catNB_fuser(onehot_pred_emg, onehot_pred_eeg, targets)
    return fuser, onehot

def function_fuse_LOO(args):
    start=time.time()
    emg_set_path=args['emg_set_path']
    eeg_set_path=args['eeg_set_path']
    
    emg_set=ml.pd.read_csv(emg_set_path,delimiter=',')
    eeg_set=ml.pd.read_csv(eeg_set_path,delimiter=',')
    
    emg_set,eeg_set=balance_set(emg_set,eeg_set)
    
    eeg_masks=get_ppt_split(eeg_set,args)
    emg_masks=get_ppt_split(emg_set,args)
    
    accs=[]
    emg_accs=[] #https://stackoverflow.com/questions/13520876/how-can-i-make-multiple-empty-lists-in-python
    eeg_accs=[]
    
    f1s=[]
    emg_f1s=[]
    eeg_f1s=[]
    
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
        else:
            emg_others=ml.drop_ID_cols(emg_others)
            eeg_others=ml.drop_ID_cols(eeg_others)
            emg_model,eeg_model=train_models_opt(emg_others,eeg_others,args)
        
        classlabels = emg_model.classes_
        
        emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            
        #targets, predlist_emg, predlist_eeg, predlist_fusion = synced_predict(emg_ppt, eeg_ppt, emg_model, eeg_model, classlabels,args)
        targets, predlist_emg, predlist_eeg, predlist_fusion = refactor_synced_predict(emg_ppt, eeg_ppt, emg_model, eeg_model, classlabels,args)
        
        if args['fusion_alg']=='bayes':
            fuser,onehotEncoder=train_bayes_fuser(emg_model,eeg_model,emg_train_split_fusion,eeg_train_split_fusion,classlabels,args)
            predlist_fusion=fusion.bayesian_fusion(fuser,onehotEncoder,predlist_emg,predlist_eeg,classlabels)
        
        #acc_emg,acc_eeg,acc_fusion=evaluate_results(targets, predlist_emg, correctness_emg, predlist_eeg, correctness_eeg, predlist_fusion, correctness_fusion, classlabels)
        
        gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels=classes_from_preds(targets,predlist_emg,predlist_eeg,predlist_fusion,classlabels)
        '''could calculate log loss if got the probabilities back''' #https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
        
        #plot_confmats(gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels)
        
        #acc_emg=accuracy_score(gest_truth,gest_pred_emg)
        #acc_eeg=accuracy_score(gest_truth,gest_pred_eeg)
        #acc_fusion=accuracy_score(gest_truth,gest_pred_fusion)
        
        #f1_emg=f1_score(gest_truth,gest_pred_emg,average='weighted')
        #f1_eeg=f1_score(gest_truth,gest_pred_eeg,average='weighted')
        #f1_fusion=f1_score(gest_truth,gest_pred_fusion,average='weighted')
        
        #kappa=cohen_kappa_score(gest_truth,gest_pred_fusion)
        
        emg_accs.append(accuracy_score(gest_truth,gest_pred_emg))
        eeg_accs.append(accuracy_score(gest_truth,gest_pred_eeg))
        accs.append(accuracy_score(gest_truth,gest_pred_fusion))
        
        emg_f1s.append(f1_score(gest_truth,gest_pred_emg,average='weighted'))
        eeg_f1s.append(f1_score(gest_truth,gest_pred_eeg,average='weighted'))
        f1s.append(f1_score(gest_truth,gest_pred_fusion,average='weighted'))
        
        kappas.append(cohen_kappa_score(gest_truth,gest_pred_fusion))
        
    mean_acc=stats.mean(accs)
    median_acc=stats.median(accs)
    mean_emg=stats.mean(emg_accs)
    mean_eeg=stats.mean(eeg_accs)
    mean_f1_emg=stats.mean(emg_f1s)
    mean_f1_eeg=stats.mean(eeg_f1s)
    mean_f1_fusion=stats.mean(f1s)
    median_f1=stats.median(f1s)
    median_kappa=stats.median(kappas)
    end=time.time()
    #return 1-mean_acc
    return {
        'loss': 1-median_kappa,
        'status': STATUS_OK,
        'fusion_mean':mean_acc,
        'fusion_median':median_acc,
        'emg_mean_acc':mean_emg,
        'eeg_mean_acc':mean_eeg,
        'emg_f1_mean':mean_f1_emg,
        'eeg_f1_mean':mean_f1_eeg,
        'fusion_f1_mean':mean_f1_fusion,
        'elapsed_time':end-start,}

def plot_opt_in_time(trials):
    fig,ax=plt.subplots()
    ax.plot(range(1, len(trials) + 1),
            [1-x['result']['loss'] for x in trials], 
        color='red', marker='.', linewidth=0)
    #https://www.kaggle.com/code/fanvacoolt/tutorial-on-hyperopt?scriptVersionId=12981074&cellId=97
    ax.set(title='accuracy over time')
    plt.show()
    
def plot_stat_in_time(trials,stat,ylower=0,yupper=1,showplot=True):
    fig,ax=plt.subplots()
    ax.plot(range(1, len(trials) + 1),
            [x['result'][stat] for x in trials], 
        color='red', marker='.', linewidth=0)
    #https://www.kaggle.com/code/fanvacoolt/tutorial-on-hyperopt?scriptVersionId=12981074&cellId=97
    ax.set(title=stat+' over time')
    ax.set_ylim(ylower,yupper)
    if showplot:
        plt.show()
    return fig
    
    # Plot something showing which were which models?
    # eg with vertical fill
    # https://stackoverflow.com/questions/23248435/fill-between-two-vertical-lines-in-matplotlib

    
def setup_search_space():
    space = {
            'emg':hp.choice('emg model',[
                {'emg_model_type':'RF',
                 'n_trees':scope.int(hp.quniform('emg.RF.ntrees',10,50,q=10)),
                 #integerising search space https://github.com/hyperopt/hyperopt/issues/566#issuecomment-549510376
                 },
                {'emg_model_type':'kNN',
                 'knn_k':scope.int(hp.quniform('emg.knn.k',1,5,q=1)),
                 },
                {'emg_model_type':'LDA',
                 'LDA_solver':hp.choice('emg.LDA_solver',['svd','lsqr','eigen']),
                 'shrinkage':hp.uniform('emg.lda.shrinkage',0.0,1.0),
                 },
                {'emg_model_type':'QDA',
                 'regularisation':hp.uniform('emg.qda.regularisation',0.0,1.0), #https://www.kaggle.com/code/code1110/best-parameter-s-for-qda/notebook
                 },
            #    {'emg_model_type':'SVM',
             #    'svm_C':hp.uniform('emg.svm.c',0.1,100),
              #   }
                ]),
            'eeg':hp.choice('eeg model',[
                {'eeg_model_type':'RF',
                 'n_trees':scope.int(hp.quniform('eeg_ntrees',10,50,q=10)),
                 },
                {'eeg_model_type':'kNN',
                 'knn_k':scope.int(hp.quniform('eeg.knn.k',1,5,q=1)),
                 },
                {'eeg_model_type':'LDA',
                 'LDA_solver':hp.choice('eeg.LDA_solver',['svd','lsqr','eigen']),
                 'shrinkage':hp.uniform('eeg.lda.shrinkage',0.0,1.0),
                 },
                {'eeg_model_type':'QDA',
                 'regularisation':hp.uniform('eeg.qda.regularisation',0.0,1.0),
                 },
             #   {'eeg_model_type':'SVM',
              #   'svm_C':hp.uniform('eeg.svm.c',0.1,100),
                 # naming convention https://github.com/hyperopt/hyperopt/issues/380#issuecomment-685173200
              #   }
                ]),
            'fusion_alg':hp.choice('fusion algorithm',[
                'mean',
                '3_1_emg',
                '3_1_eeg',
                'bayes']),
            #'emg_set_path':params.emg_set_path_for_system_tests,
            #'eeg_set_path':params.eeg_set_path_for_system_tests,
            'emg_set_path':params.emg_waygal,
            'eeg_set_path':params.eeg_waygal,
            'using_literature_data':True,
            }
    return space

def optimise_fusion():
    space=setup_search_space()
    trials=Trials() #http://hyperopt.github.io/hyperopt/getting-started/minimizing_functions/#attaching-extra-information-via-the-trials-object
    best = fmin(function_fuse_LOO,
                space=space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials)
    return best, space, trials
    

if __name__ == '__main__':
    
    #space=stochastic.sample(setup_search_space())
    #best_results=function_fuse_LOO(space)
    
    
    best,space,trials=optimise_fusion()
    best_results=function_fuse_LOO(space_eval(space,best))
    #could just get trials.results?
    
    #print(best)
    print(space_eval(space,best))
    print(1-(best_results['loss']))
    emg_acc_plot=plot_stat_in_time(trials, 'emg_mean_acc')
    eeg_acc_plot=plot_stat_in_time(trials, 'eeg_mean_acc')
    #plot_stat_in_time(trials, 'loss')
    fus_f1_plot=plot_stat_in_time(trials,'fusion_f1_mean')
    #plot_stat_in_time(trials,'elapsed_time',0,200)
    
    table=pd.DataFrame(trials.trials)
    table_readable=pd.concat(
        [pd.DataFrame(table['result'].tolist()),
         pd.DataFrame(pd.DataFrame(table['misc'].tolist())['vals'].values.tolist())],
        axis=1,join='outer')
    
    
    #print('plotting ppt1 just to get a confmat')
    #ppt1acc=function_fuse_pptn(space_eval(space,best),1,plot_confmats=True)
    
    
    '''PICKLING THE TRIALS OBJ'''
    
    currentpath=os.path.dirname(__file__)
    result_dir=params.waygal_results_dir
    resultpath=os.path.join(currentpath,result_dir)
    
    trials_obj_path=os.path.join(resultpath,'trials_obj.p')
    pickle.dump(trials,open(trials_obj_path,'wb'))
    
    '''saving figures of performance over time'''
    emg_acc_plot.savefig(os.path.join(resultpath,'emg_acc.png'))
    eeg_acc_plot.savefig(os.path.join(resultpath,'emg_acc.png'))
    fus_f1_plot.savefig(os.path.join(resultpath,'fusion_f1.png'))
    
    #load_trials_var=pickle.load(open(filename,'rb'))
    
    
    #for properly evaluating results later: https://towardsdatascience.com/multiclass-classification-evaluation-with-roc-curves-and-roc-auc-294fd4617e3a
    
    raise KeyboardInterrupt('ending execution here!')
    
    
    
    '''per_ppt_accs = ml.pd.DataFrame(list(zip(pptIDs,emg_accs,eeg_accs,fus_accs)),columns=['pptID','emg_acc','eeg_acc','fusion_acc'])'''
    