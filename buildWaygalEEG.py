#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:45:43 2023

@author: michael
"""
import params as params
import handleFeats as feats

dataset='jeongSyncCSP'

skipfails=True
#period=1
period=1000
datatype='eeg'

if dataset=='waygal':

    directory_path=params.all_channel_waygal_EEG
    output_file=params.eeg_32_waygal
    
elif dataset=='jeong':
    directory_path=params.jeong_EEGdir
    output_file=params.eeg_jeong_feats
    
elif dataset=='jeongCSP':
    directory_path=params.jeongCSP_EEGdir
    output_file=params.eeg_jeongCSP_feats
    
elif dataset=='jeongSyncCSP':
    directory_path=params.jeongSyncCSP_EEGdir
    output_file=params.eeg_jeongSyncCSP_feats
    
elif dataset=='jeongEMG':
    directory_path=params.jeong_EMGdir
    output_file=params.jeong_EMGfeats
    
else:
     raise ValueError('I don\'t know what dataset you mean by '+dataset)   

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