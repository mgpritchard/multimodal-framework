#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 21:33:38 2021

@author: pritcham

module to contain functionality related to offline operation & testing
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename
from live_feature_extraction import calc_feature_vector
from generate_training_matrix import gen_training_matrix
from labelstoClass import numtoClass
import time
import scipy
from handleML import *

def split_datasets(toggle='single'):
    homepath=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    Tk().withdraw()
    
    if toggle=='single':
        title='select emg file to split'
        datafile=askopenfilename(initialdir=homepath,title=title,filetypes = (("csv files","*.csv"),("all files","*.*")))
    elif toggle=='multi':
        title='select emg files to split'
        datafiles=askopenfilenames(initialdir=homepath,title=title,filetypes = (("csv files","*.csv"),("all files","*.*")))
    
    title_h1='location for half 1'
    title_h2='location for half 2'
    half1dest=askdirectory(initialdir=homepath,title=title_h1)
    half2dest=askdirectory(initialdir=homepath,title=title_h2)
    
    if toggle=='single':
        split_emg(datafile,half1dest,half2dest)
    elif toggle=='multi':
        for file in datafiles:
            split_emg(file,half1dest,half2dest)
        

def split_emg(file,half1loc,half2loc):
    emg=pd.read_csv(file,delimiter=",")
    emg_half1=emg[['Timestamp','EMG1','EMG2','EMG3','EMG4']].copy()
    emg_half2=emg[['Timestamp','EMG5','EMG6','EMG7','EMG8']].copy()
    emg_name=os.path.basename(file)
    half1_name=os.path.join(half1loc,emg_name)
    half2_name=os.path.join(half2loc,emg_name)
    emg_half1.to_csv(half1_name,index=False)
    emg_half2.to_csv(half2_name,index=False)
    
def gen_train_set():
    homepath=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    Tk().withdraw()
    title='directory of time series data'
    set_dir=askdirectory(title=title,initialdir=homepath)
    title_outp='output dataset file'
    output_name=asksaveasfilename(title=title_outp,initialdir=homepath)
    gen_training_matrix(set_dir, output_name, cols_to_ignore=None)
    labelled_file=output_name[:-4]+'Class.csv'
    numtoClass(output_name,labelled_file)

def matrix_from_csv_file(file):
    csv_data=pd.read_csv(file,delimiter=",").values
    matrix = csv_data[1:]
    headers = csv_data[0]
    print ('MAT', (matrix.shape))
	#print ('HDR', (headers.shape))
    return matrix, headers

def loadset(label):
    path=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    title='select time series data for '+label
    Tk().withdraw()
    datafile=askopenfilename(initialdir=path,title=title,filetypes = (("csv files","*.csv"),("all files","*.*")))
    data=matrix_from_csv_file(datafile)[0]
    _,state,_=os.path.basename(datafile)[:-4].split('-')
    return data, state

def sync_crop(data1,data2):
    data1,data2=syncstarts(data1,data2)
    data1,data2=syncends(data1,data2)
    return data1,data2
    
def syncstarts(data1,data2):
    starts=[data1[0,0],data2[0,0]]
    latest = np.argmax(starts)
    start = starts[latest]
    data1=crop_to_start(data1,start)
    data2=crop_to_start(data2,start)
    return data1,data2
    
def syncends(data1,data2):
    ends=[data1[-1,0],data2[-1,0]]
    earliest = np.argmin(ends)
    end = ends[earliest]
    data1=crop_to_end(data1,end)
    data2=crop_to_end(data2,end)
    return data1,data2
    
def crop_to_start(data,start):
    startrow=np.argmax(data[:,0]>=start)
    cropped=data[startrow:,:]
    return cropped

def crop_to_end(data,end):
    endrow=np.argmax(data[:,0]>=end)
    cropped=data[:endrow,:]
    return cropped

def get_time_slice(full_matrix, start = 0., period = 1.):
    """
    Returns a slice of the given matrix, where start is the offset and period is 
    used to specify the length of the signal. 
    Parameters:
        full_matrix (numpy.ndarray): matrix returned by matrix_from_csv()
        start (float): start point (in seconds after the beginning of records) 
        period (float): duration of the slice to be extracted (in seconds)
    Returns:
        numpy.ndarray: 2D matrix with the desired slice of the matrix
        float: actual length of the resulting time slice    
    Author:
        Original: [lmanso]
        Reimplemented: [fcampelo]
    """   
    # Changed for greater efficiency [fcampelo]
    rstart  = full_matrix[0, 0] + start
    index_0 = np.max(np.where(full_matrix[:, 0] <= rstart))
    index_1 = np.max(np.where(full_matrix[:, 0] <= rstart + period))
    
    duration = full_matrix[index_1, 0] - full_matrix[index_0, 0]
    return full_matrix[index_0:index_1, :], duration

def get_feats(vect):
    nsamples=150
    resampled, rx = scipy.signal.resample(vect[:,:], num = nsamples, 
                                 t = vect[:, 0], axis = 0)
    feats,headers=calc_feature_vector(resampled,None)
    doubleup=0
    if doubleup:
        featsvec=np.append(feats,feats)
        #delete some duplicated columns: #specific to 8channel emg
        del_ind=[32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,
             47,88,89,90,91,92,93,94,95,160,161,162,163,164,
             165,166,167,168,169,170,171,172,173,174,175,
             216, 217,218,219,220,221,222,223,256,257,258,
             259,260,261,262,263,264,265,266,267,268,269,
             270,271,312,313,314,315,316,317,318,319]
        featsvec=np.delete(featsvec,del_ind)
    else:
        featsvec=feats
    values=np.nan_to_num(featsvec)
    values=values.reshape(1,-1)
    return values

def slice_and_predict(data,offset,period,model,verbose):
    current_slice,duration=get_time_slice(data,offset,period)
    if verbose:
        print('slice: ',duration)
    current_feats=get_feats(current_slice[:,1:])
    current_pred = predict_from_array(model,current_feats)
    current_distro = prob_dist(model,current_feats)
    return current_slice,current_pred,current_distro 