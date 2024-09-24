#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:45:43 2023

@author: michael
"""

''' Script for feature-extracting ANY dataset, from a folder of raw biosignal recordings to a single CSV '''
''' also functionality to split a featureset into individual Holdout subjects & a "remainder" Development set'''

import params as params
import handleFeats as feats
import pandas as pd



def split_holdout(full_path,holdoutIDs,save_path_stem):
    fullset = pd.read_csv(full_path,delimiter=',')
    #save_dir=os.path.dirname(save_path_stem)
    
    for HO in holdoutIDs:
        HOset = fullset[fullset['ID_pptID']==HO]
        HOset.to_csv((save_path_stem+'_ppt'+str(HO)+'.csv'),
                     sep=',',index=False,float_format="%.18e")
        
    devset = fullset[~fullset['ID_pptID'].isin(holdoutIDs)]
    devset.to_csv((save_path_stem+'_noHO.csv'),
                  sep=',',index=False,float_format="%.18e")
    


def get_paths(dataset):

    if dataset=='waygal':
        directory_path=params.all_channel_waygal_EEG
        output_file=params.eeg_32_waygal
        
    elif dataset=='jeong':              # initial jeong EEG
        directory_path=params.jeong_EEGdir
        output_file=params.eeg_jeong_feats
        
    elif dataset=='jeongCSP':           # jeong EEG with CSP
        directory_path=params.jeongCSP_EEGdir
        output_file=params.eeg_jeongCSP_feats
        
    elif dataset=='jeongSyncCSP':       # jeong EEG with CSP, timesynced to EMG
        directory_path=params.jeongSyncCSP_EEGdir
        output_file=params.eeg_jeongSyncCSP_feats
        
    elif dataset=='jeongEMG':           # jeong EMG, timesynced to EEG
        directory_path=params.jeong_EMGdir
        output_file=params.jeong_EMGfeats
        
    elif dataset=='jeongSyncRawEEG':    # jeong EEG no CSP, timesynced to EMG
        directory_path=params.jeongSyncRawEEGdir
        output_file=params.eeg_jeongSyncRaw_feats
        
    elif dataset=='jeongSyncWideband_noCSP':    #jeong EEG no CSP, synced, wider bandpass filter
        directory_path=params.jeong_noCSP_WidebandDir
        output_file=params.jeong_noCSP_WidebandFeats
        
    elif dataset == 'jeongFiltFiltEEG':         # jeong EEG, trialling zero-phase butterworth
        directory_path=params.jeong_eeg_filtfilt_dir
        output_file=params.jeong_eeg_filtfilt_path
        
    else:
         raise ValueError('I don\'t know what dataset you mean by '+dataset)
         
    return directory_path,output_file


def extract_featureset(directory_path, output_file, skipfails=True, period=1000, datatype='eeg'):
    print('Are the following parameters OK?')
    print('Skipfails: ',skipfails,'\n',
          'Period: ',period,'\n',
          'datatype: ',datatype,'\n',
          'Data directory: ',directory_path,'\n',
          'Dataset location: ',output_file)
    
    go_ahead=input('Ready to proceed [Y/N] ')
    
    if go_ahead=='Y' or go_ahead=='y':
        feats.make_feats(directory_path, output_file, datatype, period, skipfails)
    else:
        print('aborting...')
    
    
if __name__ == '__main__':
    if 1:
        holdoutIDs=[1,6,11,16,21]
        full_path = r'H:/Jeong11tasks_data/jeong_FiltFilt_EEG_feats.csv'
        split_holdout(full_path,holdoutIDs,save_path_stem=r"H:\Jeong11tasks_data\jeong_FiltFilt_EEG")
    
    
    if 0:
        #dataset='jeongSyncWideband_noCSP' # final one for use in experiments
        dataset = 'jeongFiltFiltEEG'       # testing with zero-phase filter
        
        skipfails=True
        #period=1
        period=1000
        datatype='eeg'
        
        directory_path,output_file = get_paths(dataset)
        
        extract_featureset(directory_path, output_file, skipfails, period, datatype)
        
        if 0:
            holdoutIDs=[1,6,11,16,21]
            full_path = output_file#[:-4] + '_Labelled.csv' #NO, should be numerical labels here
            save_path_stem = output_file[:-4]
            split_holdout(full_path,holdoutIDs,save_path_stem=r"H:\Jeong11tasks_data\jeong_FiltFilt_EEG")
    
    