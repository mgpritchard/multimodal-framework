#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 21:17:15 2022

@author: pritcham
"""

import zipfile
import py7zr
import os, sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from handleDatawrangle import Rawfile

def get_trials_to_merge():
    triallist=input('input trials to merge into a set, separated by hyphen eg 0-a-b.\n If all merging is done, enter Done: ')
    try:
        trials=triallist.split('-')
    except:
        trials = triallist
    return trials

def list_recordings_from_zip(pptzipfile):
    if pptzipfile[-3:]=='zip':
        pptzip=zipfile.ZipFile(pptzipfile)
        namelist=pptzip.namelist()
    elif pptzipfile[-2:]=='7z':
        pptzip=py7zr.SevenZipFile(pptzipfile)
        namelist=pptzip.getnames()
    else:
        print('Error: needs to be a .zip or .7z')
        return
        
    filelist = []
    for filename in namelist:
        # Ignore non-CSV files
        if not filename.lower().endswith('.csv'):
            continue
        # For safety we'll ignore files containing the substring "test". 
        # [Test files should not be in the dataset directory in the first place]
        if 'test' in filename.lower():
            continue
        # We don't want to double-count so are ignoring the eeg files and
        # just counting up the emg
        if 'eeg' in filename.lower():
            continue
        try:
            name, state, count = filename[:-4].split('-')
        except:
            print ('Wrong file name', filename)
            sys.exit(-1)
        recording=Rawfile(filename,name,state,count)
        filelist.append(recording)
    return filelist

def merge_trials(filelist):
    trials=[]
    run=0
    while trials != ['Done']:
        trials=get_trials_to_merge()
        run+=1
        for file in filelist:
            try:
                filetrial = file.ppt[-1]
            except:
                filetrial = str(file.ppt)
            if not filetrial.isalpha():
                filetrial = run
            if filetrial in trials:
                filetrial = run
            file.ppt=filetrial
    return filelist

def label_gesture_trials(filelist):
    for idx,file in enumerate(filelist):
        trial = str(file.ppt[-1])
        if not trial.isalpha():
            trial = 0
        filelist[idx] = [file,trial]
    return filelist

def count_gestures(filelist,gestlist):
    triallist=list(set([str(file[1]) for file in filelist]))
    print(filelist[0][0].ppt)
    for trial in sorted(triallist):
        thesetrials = [file[0].label for file in filelist if str(file[1])==trial]
        print('\ntrial '+str(trial))
        for gest in gestlist:
            gestcount = thesetrials.count(gest)
            print(gest+': '+str(gestcount),end=" | ")
              
if __name__ == '__main__':
    homepath=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    Tk().withdraw()
    title='zip file of ppt data to quantify' 
    pptzipfile=askopenfilename(title=title,initialdir=homepath)
    files=list_recordings_from_zip(pptzipfile)
    #files=merge_trials(files)
    files_with_labels=label_gesture_trials(files)
    gestlist=['close','open','neutral','grasp','lateral','tripod']
    count_gestures(files_with_labels,gestlist)
