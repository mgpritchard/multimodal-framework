#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:45:43 2023

@author: michael
"""
import params as params
import handleFeats as feats

skipfails=True
period=1
datatype='eeg'

directory_path=params.all_channel_waygal_EEG
output_file=params.eeg_32_waygal

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