#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:52:21 2022

@author: michael
"""

#Learn how to use pytest https://docs.pytest.org/en/6.2.x/ as a way of making
# this work - will avoid restarting at every failure?

import os
import handleDatawrangle as dw
import numpy as np

here=os.path.dirname(os.path.realpath(__file__))
rawdir= here + '/testbench/emg/raw'
cropdir= here + '/testbench/emg/crop'
procdir= here + '/testbench/emg/proc'

generated_raw_dir=dw.get_dir('**navigate to testbench raw emg**')
assert generated_raw_dir==rawdir, \
    f"expecting directory to be: {rawdir}, \ninstead got: {generated_raw_dir}"
print('fetching directory ok')

raw_list=dw.list_raw_files(rawdir)
rawpath1=rawdir+'/dummy-noclass-1.csv'
rawpath2=rawdir+'/dummy-noclassinverted-1.csv'
rawpath3=rawdir+'/dummy-noclasscopy-1.csv'
correct_raw_list = [dw.Rawfile(rawpath1,'dummy','noclass','1'),dw.Rawfile(rawpath2,'dummy','noclassinverted','1'),dw.Rawfile(rawpath3,'dummy','noclasscopy','1')]
assert raw_list == correct_raw_list, \
    f"expecting raw file list to read:\n {correct_raw_list}\ninstead got:\n {raw_list}"
print('collecting datafile list ok')

rawdir_eeg=here + '/testbench/eeg/raw'
cropdir_eeg=here + '/testbench/eeg/crop'
raw_eeg_list=dw.list_raw_files(rawdir_eeg)
raw_emg_list=raw_list
cropdir_emg=cropdir
dw.sync_raw_files(raw_emg_list,raw_eeg_list,cropdir_emg,cropdir_eeg,approval_required=0)
print('successfully skipped emg with no eeg match')

emg=np.genfromtxt(cropdir_emg+'/dummy-noclass-1.csv', delimiter = ',')
eeg=np.genfromtxt(cropdir_eeg+'/dummy-noclass-1.csv', delimiter = ',')
starts=[emg[0,0],eeg[0,0]]
latest = np.argmax(starts)
start = starts[latest]
assert emg[0,0]>=start, \
    f"expected emg start {emg[0,0]} to be >= {start} which is latest of {starts}"
assert eeg[0,0]>=start, \
    f"expected eeg start {eeg[0,0]} to be >= {start} which is latest of {starts}"
print(f'starts synced successfully: [{starts}]')

ends=[emg[-1,0],eeg[-1,0]]
earliest = np.argmin(ends)
end = ends[earliest]
assert emg[-2,0]<=end, \
    f"expected emg end-1 {emg[-2,0]} to be <= {end} which is earliest of {ends}"
assert eeg[-2,0]<=end, \
    f"expected eeg end-1 {eeg[-2,0]} to be <= {end} which is earliest of {ends}"
print(f'ends synced successfully: [{ends}]')

##EMG has 13 digits of UNIX in col 0. EEG has 10 digits + 5 dp of UNIX in col R
##EMG has 8 cols 1-9 of emg data
##EEG has 8 cols 0-7 of eeg data, 3 cols 8=10 of accel data...
#           3 cols 11-13 of gyro data, 1 col 14 of battery, 1 col 15 of package num...
#           1 col 16 of [other], 1 col 17 of UNIX, 1 col 18 of marker
#    print(BoardShim.get_other_channels(8))
    

