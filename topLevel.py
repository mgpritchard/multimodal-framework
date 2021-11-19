#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 22:08:53 2021

@author: pritcham

top level script of a multimodal framework
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory
from queue import Queue
import logging
from threading import Thread
import time
from handleEMG import *
from handleML import *
from handleOffline import *
from handleFusion import *
import pickle
import plot_dists as proto_plot

def liveclassify():
    threadEMG=Thread(target=handleEMG.read_emg(),daemon=True)
    threadEEG=Thread(target=handleEMG.read_emg(),daemon=True)
    threadML=Thread(target=handleML.classify_continuous(),daemon=True)
    threadEMG.start()
    threadEEG.start()
    threadML.start()
    return

def onlineclassify():
    path=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model1=load_model('mode 1',path)
    model2=load_model('mode 2',path)
    #RF model might be less erratic with conf interval?
    #https://github.com/scikit-learn-contrib/forest-confidence-interval#readme
    #seems like it cna help it to get a better prob dist? investigate
    
    time.sleep(1)
    mode1arr,mode2arr,fusionarr=pyoc_fuse_emgonly(model1,model2,'dupe',outp='window',limit=105)
    #mode1arr,mode2arr,fusionarr=pyoc_fuse_adapt_autocorr(model1,model2,'dupe',outp='window',limit=105,autocorr_lag=10)
    return mode1arr,mode2arr,fusionarr,model1.classes_

    #maybe every x predictions do a full recalibrate. prompt them with
    #the gestures, measure how accurate the weightings are for *each* gesture
    #with JS (as it measures stability over time), compute an averaged weight
    #from the reliability measures of all gestures, update weights accordingly
    #could even retrain classifiers in background after the calib
    
    #as in it goes "emg was very stable over close, and tripod, but wobbled
    #with open, so it's avg reliabilty is x. meanwhile eeg is unstable on
    #lateral but very stable on grasp and close, so its avg reliability...
    
    #fuck you could theoretically have a neural net continually learnig the
    #best weighting scheme

def offlineclassify(toggle_report):
    #model_name = 'EMG-GNB-SYNTH-CALIB.sav'  #load GPT2-augmented model
    #model_name='test-model.sav'            #load model without GPT2
    path=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model1=load_model('mode 1',path)
    model2=load_model('mode 2',path)
    classlabels=model1.classes_
    mode1,state=loadset('first mode')
    mode2,_=loadset('second mode')
    quickplot(mode1,'mode 1 before')
    quickplot(mode2,'mode 2 before')
    mode1,mode2=sync_crop(mode1,mode2)
    quickplot(mode1,'mode 1 after')
    quickplot(mode2,'mode 2 after')
    offset=0
    period=1
    #pred_array1=[] #this approach etc used previously when separate arrs
    colnames=['Mode 1','Mode 2','Fusion','Weighting 1','Weighting 2']
    preds={col: [] for col in colnames}
    while (offset <= (mode1[-1,0]-mode1[0,0])-period):
        #print('start: %f' % time.time())
        tstart=time.time()
        
        current_slice1,pred1,distro1=slice_and_predict(mode1,offset,period,model1,0)
        current_slice2,pred2,distro2=slice_and_predict(mode2,offset,period,model2,0)
        
        gesture1=pred_gesture(pred1,0)
        gesture2=pred_gesture(pred2,0)
        #pred_array1.append(str(gesture1[0])) #this approach previously used
        preds["Mode 1"].append(str(gesture1[0]))
        preds["Mode 2"].append(str(gesture2[0]))
        
        mean_distro=fuse_mean(distro1,distro2)
        gesture_mean=pred_from_distro(classlabels,mean_distro)
        preds["Fusion"].append(str(gesture_mean))
        
        w1_distro=fuse_linweight(distro1,distro2,75,25) #bundle these lines?
        gesture_w1=pred_from_distro(classlabels,w1_distro)
        preds["Weighting 1"].append(str(gesture_w1))
        
        w2_distro=fuse_linweight(distro1,distro2,25,75)
        gesture_w2=pred_from_distro(classlabels,w2_distro)
        preds["Weighting 2"].append(str(gesture_w2))
        
        print(time.time(),'\n','Mode 1: ',gesture1,'\n','Mode 2: ',gesture2,'\n','Fusion: ',gesture_mean)#,'\n-------------')
        print('Weighted to 1: ',gesture_w1,'\nWeighted to 2: ',gesture_w2)
        print('-------------')
        
        offset+=0.5*period
        #quickplot(current_slice,'mode 1 slice n')
        tend=time.time()
        #print('end: %f'% tend)
        #print('delta: %f'% (tend-tstart))
        while (tend-tstart < (period/2)):
            time.sleep(0.1)
            tend=time.time()
    quickplot(current_slice1,'mode 1 slice last')
    
    if toggle_report:
        state=state.capitalize()
        #scores=eval_multi([pred_array1,pred_array2,pred_arrayfusion,pred_array_w1,pred_array_w2],state) #make these all one array!!
        scores=eval_multi(list(preds.values()),state)
        print(scores)
        acc = pd.DataFrame([scores],columns=colnames)
        #result = pd.DataFrame(list(zip(pred_array1,pred_array2,pred_arrayfusion,pred_array_w1,pred_array_w2)),columns=['Mode 1','Mode 2','Fusion','Mostly 1','Mostly 2'])
        result = pd.DataFrame.from_dict(preds)
        result.columns=colnames
        return result, acc
    
    return

def quickplot(data,label):
    fig,ax=plt.subplots()
    ax.plot(data[:,0],data[:,1])
    ax.set(title=label)
    plt.show()

if __name__ == '__main__':
    
    #ALLOW parameters passed from commandline, but if CLI inputs are none
    #THEN ask for user input instead
    
    onlinemode=1#int(input('Classifying live? 1:Y 0:N '))
    print(onlinemode)
    
    #IDEAL could be a basic gui unless you CLI override to use the text input
    #method. would have for eg tickboxes to select which sensing modes you are
    #using, what kinda model, if you want to train or calibrate or just go, etc
    
    #fusion approaches etc then PROBABLY need to be like... objects with
    #attributes and names and overloaded functions so its just
    # fusionmodel.fuse but fusionmodel could = mean or = JS or = conf etc
    
    if not onlinemode:
        results,scores=offlineclassify(1)
        #split_datasets('multi')
        #gen_train_set()
        #train_offline()
    
    else:
        mode1arr,mode2arr,fusionarr,classlist=onlineclassify()
        proto_plot.plotdist(fusionarr,classlist)
        proto_plot.plotdist(mode1arr,classlist)
        proto_plot.plotdist(mode2arr,classlist)
        plt.show()
        
        '''should start to actually measure accuracy and use ROC extended
        to multiclass with one-vs-rest as per
        https://scikit-learn.org/0.15/auto_examples/plot_roc.html'''
       
        #liveclassify()
        '''
        path=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        model1=load_model('mode 1',path)
        results=pyoc_slicetester(model1)
        '''
        #class labels might be wrong?

    #https://stackoverflow.com/questions/47096507/reducing-the-number-of-arguments-in-function-in-python