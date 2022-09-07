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

def select_feats(featureset,alg=None):
    print('\n\n*no feature selection currently implemented*\n\n')
    return featureset

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

def make_feats(directory_path=None, output_file=None, datatype="",period=1000):
    if directory_path is None:
        directory_path=ask_for_dir(datatype)
    if output_file is None:
        output_file=ask_for_savefile(datatype)
    featset=genfeats.gen_training_matrix(directory_path, output_file, cols_to_ignore=None, singleFrame=0,period=period)
    return featset
    
def feats_pipeline():
    make_feats(datatype='emg')
    make_feats(datatype='eeg')
    
if __name__ == '__main__':
    emg_data_path='/home/michael/Documents/Aston/MultimodalFW/repo/multimodal-framework/testbench/emg/dupes_removed'
    emg_feats_file='/home/michael/Documents/Aston/MultimodalFW/repo/multimodal-framework/testbench/emg/emg_test_feats.csv'
    make_feats(directory_path=emg_data_path,output_file=emg_feats_file,datatype='emg')