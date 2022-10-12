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
    datain=here+'/testbench/emg/data_with_dupes'
    dataout=here+'/testbench/emg/dupes_removed'
    wrangle.process_emg(datain,dataout)
    #sync_raw_files()    #not yet ready as EEG data not ready
    emg_data_path=here+'/testbench/emg/dupes_removed'
    emg_feats_file=here+'/testbench/emg/emg_test_feats.csv'
    feats.make_feats(directory_path=emg_data_path,output_file=emg_feats_file,datatype='emg')

