#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:15:27 2022

@author: pritcham

module to contain functionality related to wrangling & processing of data

"""
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfile
import generate_training_matrix as feats

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
    savefile=asksaveasfile(title=title,initialdir=homepath)
    return savefile

def make_feats(directory_path=None, output_file=None, datatype=""):
    if directory_path is None:
        directory_path=ask_for_dir(datatype)
    if output_file is None:
        output_file=ask_for_savefile()
    feats.gen_training_matrix(directory_path, output_file, cols_to_ignore=None, singleFrame=0)
    
def feats_pipeline():
    make_feats(datatype='emg')
    make_feats(datatype='eeg')