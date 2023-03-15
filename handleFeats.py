#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:15:27 2022

@author: pritcham

module to contain functionality related to wrangling & processing of data

"""
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename
import generate_training_matrix as genfeats
from sklearn.feature_selection import SelectPercentile, f_classif

def select_feats(featureset,alg=None):
    print('\n\n*no feature selection currently implemented*\n\n')
    #look into MRMR?
    return featureset

def sel_percent_feats_df(data,percent=15):
    #https://stackoverflow.com/questions/39839112/the-easiest-way-for-getting-feature-names-after-running-selectkbest-in-scikit-le
    target=data['Label']
    attribs=data.drop(columns=['Label'])
    selector=SelectPercentile(f_classif,percentile=percent)
    selector.fit(attribs,target)
    col_idxs=selector.get_support(indices=True)
    #selected=attribs.iloc[:col_idxs]
    #selected['Label']=target
    return col_idxs
    

def ask_for_dir(datatype=""):
    homepath=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    Tk().withdraw()
    title='directory of '+datatype+' data for feature extraction' 
    set_dir=askdirectory(title=title,initialdir=homepath)
    return set_dir

def ask_for_savefile(datatype=""):
    homepath=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    Tk().withdraw()
    title='save '+datatype+' featureset as' 
    savefile=asksaveasfilename(title=title,initialdir=homepath)
    return savefile

def make_feats(directory_path=None, output_file=None, datatype="",period=1000,skipfails=False):
    '''datatype can be in any case, it does not affect mechanical functionality only cosmetic'''
    if directory_path is None:
        directory_path=ask_for_dir(datatype)
    if output_file is None:
        output_file=ask_for_savefile(datatype)
    featset=genfeats.gen_training_matrix(directory_path, output_file, cols_to_ignore=None, singleFrame=0,period=period,auto_skip_all_fails=skipfails)
    return featset
    
def feats_pipeline():
    make_feats(datatype='emg')
    make_feats(datatype='eeg')
    
if __name__ == '__main__':
    WAYGAL_P4_path='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/EEG_CSVs/'
    make_feats(directory_path=WAYGAL_P4_path,datatype='eeg',period=1,skipfails=True)
    raise
    emg_data_path='/home/michael/Documents/Aston/MultimodalFW/repo/multimodal-framework/testbench/emg/dupes_removed'
    emg_feats_file='/home/michael/Documents/Aston/MultimodalFW/repo/multimodal-framework/testbench/emg/emg_test_feats.csv'
    make_feats(directory_path=emg_data_path,output_file=emg_feats_file,datatype='emg')