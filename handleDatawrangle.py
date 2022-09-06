#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:09:41 2022

@author: pritcham

module to contain functionality related to wrangling & processing of data
"""
import os, sys
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import csv
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfile
#from queue import Queue
#import logging
#from threading import Thread
#import time
from handleEMG import *
from handleML import *
from handleOffline import sync_crop
from handleFusion import *
from topLevel import quickplot
import pickle
import generate_training_matrix as feats
import handleBFSigProc as bfsig

def get_dir(datatype='',stage=''):
    homepath=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    Tk().withdraw()
    if stage=='raw':
        title='directory of raw '+datatype+' time series data'
    elif stage== 'crop':
        title='directory for cropped raw '+datatype+' time series data'
    elif stage=='proc':
        title='directory for processed '+datatype+' time series data'
    else:
        title='directory for '+datatype+' data' 
    set_dir=askdirectory(title=title,initialdir=homepath)
    return set_dir

class Rawfile:
    def __init__(self,filepath,ppt,label,count):
        self.filepath=filepath
        self.ppt=ppt
        self.label=label
        self.count=count
        
    def __eq__(self, other): #per https://stackoverflow.com/questions/1227121/compare-object-instances-for-equality-by-their-attributes
        if not isinstance(other, Rawfile):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.filepath == other.filepath and \
                self.ppt == other.ppt and \
                    self.label == other.label and \
                        self.count == other.count
    
    def __repr__(self):
        return(f'{self.__class__.__name__}('
               f'{self.filepath!r},{self.ppt!r}, {self.label!r}, {self.count!r})')

def list_raw_files(raw_dir):
    raw_list = []
    for x in os.listdir(raw_dir):
        if not x.lower().endswith('.csv'): # Ignore non-CSV files
            continue    
        # For safety we'll ignore files containing the substring "test". 
        # [Test files should not be in the dataset directory in the first place]
        if 'test' in x.lower():
            continue
        try:
            name, state, count = x[:-4].split('-')
        except:
            if x[-7:-4]=='EEG':
                try:
                    name, state, count = x[:-9].split('-')
                except:
                    print ('Wrong file name', x)
                    sys.exit(-1)
            else:
                print ('Wrong file name', x)
                sys.exit(-1)   
        print ('Using file', x)
        full_file_path = raw_dir + '/' + x
        rawfile = Rawfile(full_file_path,name,state,count)
        raw_list.append(rawfile)
    return raw_list

def sync_raw_files(raw_emg_list,raw_eeg_list,crop_emg_dir,crop_eeg_dir,approval_required=0):
    for emgfile in raw_emg_list:
        print('working with '+repr(emgfile))
        raw_emg = matrix_from_csv_file(emgfile.filepath)
        eegfile= next((eeg for eeg in raw_eeg_list
                 if (eeg.ppt==emgfile.ppt and eeg.label==emgfile.label
                  and eeg.count == emgfile.count)),None)
        if eegfile is None:
            print('no matching eeg file... :(')
            print('skipping')
            continue
        print('found eeg: '+repr(eegfile))
        raw_eeg = matrix_from_csv_file(eegfile.filepath)
        raw_eeg=move_unicorn_time(raw_eeg)
        if approval_required:
            crop_emg,crop_eeg=approve_sync(raw_emg, raw_eeg)
        else:
            crop_emg,crop_eeg=sync_crop(raw_emg, raw_eeg)
        np.savetxt(build_path(crop_emg_dir,emgfile),crop_emg,delimiter=',')
        np.savetxt(build_path(crop_eeg_dir,eegfile),crop_eeg,delimiter=',')
        
def build_path(dirpath,file):
    filepath=dirpath+'/'+file.ppt+'-'+file.label+'-'+file.count+'.csv'
    return filepath

def approve_sync(mode1,mode2):
    quickplot(mode1,'mode 1 before')
    quickplot(mode2,'mode 2 before')
    mode1crop,mode2crop=sync_crop(mode1,mode2)
    quickplot(mode1,'mode 1 after')
    quickplot(mode2,'mode 2 after')
    approval=input('approve crop? [1/0] ')
    if approval:
        return mode1crop,mode2crop
    else:
        print('returning uncropped')
        return mode1,mode2
        
def matrix_from_csv_file(file_path):
    csv_data = np.genfromtxt(file_path, delimiter = ',')
    full_matrix = csv_data[1:]
    #headers = csv_data[0] # Commented since not used or returned [fcampelo]
    return full_matrix

def move_unicorn_time(unicorndata):
    moved=unicorndata[:,[17,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18]]
    return moved

def do_something_brainflow(eeg,sampling_rate):
    print ("\nITS NOT FINISHED YET\n")
    print('reference to average (spatial)\nBut maybe it should be Surface Laplace')
    #bfsig.plot_eeg(data,1,'raw')
    eeg[:,1:]=bfsig.ref_to_avg(eeg[:,1:])
    #bfsig.plot_eeg(data,1,'raw after ref')
    #---
    
    #bfsig.plot_fft(eeg[:,1:],sampling_rate,'FFT pre-filt')
    hp_filt_type=bfsig.FilterTypes.BUTTERWORTH.value
    hp_cutoff=0.5
    hp_order=3
    eeg=bfsig.highpass(eeg,[1,2,3,4,5,6,7,8],sampling_rate,hp_cutoff,hp_order,hp_filt_type)
    #bfsig.plot_eeg(eeg,1,'HPF')
    #bfsig.plot_fft(eeg[:,eeg_channels],sampling_rate,'FFT HPF')
    #---
    
    notch_filt_type=bfsig.FilterTypes.BUTTERWORTH.value
    notch_freq=50
    notch_width=1
    notch_order=3
    eeg=bfsig.notch(eeg,[1,2,3,4,5,6,7,8],sampling_rate,notch_freq,notch_width,notch_order,notch_filt_type)
    #bfsig.plot_eeg(data,2,'Notch')
    #bfsig.plot_fft(data[:,eeg_channels],sampling_rate,'FFT Notch')
    print('bfsig.notch')
    return eeg

def ditch_bad_columns(eeg):
    print ("\nITS NOT FINISHED YET\n")
    print ('-Deprecated in favour of achieving this when loading datafile in load_raw_brainflow-')
    eeg=eeg[0,1,2,3,4,5,6,7,8,]
    return eeg

def process_eeg(dataINdir,dataColsRearrangeddir,dataOUTdir):
    eeglist=list_raw_files(dataINdir)
    for eegfile in eeglist:
        print('processing '+repr(eegfile))
       
        eeg,sampling_rate=bfsig.load_raw_brainflow(eegfile.filepath)
        np.savetxt(build_path(dataColsRearrangeddir,eegfile),eeg,delimiter=',')
        
        eeg=do_something_brainflow(eeg,sampling_rate)
        np.savetxt(build_path(dataOUTdir,eegfile),eeg,delimiter=',')

def remove_dupe_rows(data):
    idx=0
    while idx < len(data[:-2,-1]):
        if np.array_equal(data[idx+1,:] , data[idx,:]):
            data=np.delete(data,idx+1,0)
            idx=idx-1
            if idx+1 > len(data[:-2,-1]):
                break
        idx+=1
    return data
    #seems to work but needs more thorough real data test eg compare plots
    #https://stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved
        
def process_emg(dataINdir,dataOUTdir):
    emglist=list_raw_files(dataINdir)
    for emgfile in emglist:
        print('fixing '+repr(emgfile))
        emg=matrix_from_csv_file(emgfile.filepath)
        emg=remove_dupe_rows(emg)
        np.savetxt(build_path(dataOUTdir,emgfile),emg.astype(int),fmt='%i',delimiter=',')
        
    
def data_pipeline():
    raw_emg_dir=get_dir('emg','raw')
    crop_emg_dir=get_dir('emg','crop')
    proc_emg_dir=get_dir('emg','proc')
    raw_eeg_dir=get_dir('eeg','raw')
    crop_eeg_dir=get_dir('eeg','crop')
    proc_eeg_dir=get_dir('eeg','proc')
    raw_emg_list=list_raw_files(raw_emg_dir)
    raw_eeg_list=list_raw_files(raw_eeg_dir)
    sync_raw_files(raw_emg_list,raw_eeg_list,crop_emg_dir,crop_eeg_dir,approval_required=0)
    process_emg(crop_emg_dir,proc_emg_dir)
    process_eeg(crop_eeg_dir,proc_eeg_dir)
    
if __name__ == '__main__':
    datain='/home/michael/Documents/Aston/MultimodalFW/repo/multimodal-framework/testbench/emg/data_with_dupes'
    dataout='/home/michael/Documents/Aston/MultimodalFW/repo/multimodal-framework/testbench/emg/dupes_removed'
    process_emg(datain,dataout)
        