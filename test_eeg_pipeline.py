#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:52:21 2022

@author: michael
"""

#Learn how to use pytest https://docs.pytest.org/en/6.2.x/ as a way of making
# this work - will avoid restarting at every failure?

import os
import handleDatawrangle as wrangle
import handleFeats as feats

'''https://photos.google.com/photo/AF1QipPtvjbsYyNakku1nL1zEin_bcBB3pKJ-eiepshr'''

here=os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    datain=here+'/testbench/eeg/data_from_brainflow'
    data_cols=here+'/testbench/eeg/data_rearranged_cols'
    dataout=here+'/testbench/eeg/data_after_sigproc'
    wrangle.process_eeg(datain,data_cols,dataout)
    #sync_raw_files()    #not yet ready as EEG data not ready
    eeg_data_path=here+'/testbench/eeg/data_after_sigproc'
    eeg_feats_file=here+'/testbench/eeg/eeg_test_feats.csv'
    feats.make_feats(directory_path=eeg_data_path,output_file=eeg_feats_file,datatype='eeg')
    # should the eeg feats probably be TD stat feats of the freq BPs
