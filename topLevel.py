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
import handleEMG
from handleML import *
from handleOffline import *
from handleFusion import *
import pickle

def liveclassify():
    threadEMG=Thread(target=handleEMG.read_emg(),daemon=True)
    threadEEG=Thread(target=handleEMG.read_emg(),daemon=True)
    threadML=Thread(target=handleML.classify_continuous(),daemon=True)
    threadEMG.start()
    threadEEG.start()
    threadML.start()
    return

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
    pred_array1=[]
    pred_array2=[]
    pred_arrayfusion=[]
    while (offset <= (mode1[-1,0]-mode1[0,0])-period):
        #print('start: %f' % time.time())
        tstart=time.time()
        
        current_slice1,pred1,distro1=slice_and_predict(mode1,offset,period,model1,0)
        current_slice2,pred2,distro2=slice_and_predict(mode2,offset,period,model2,0)
        
        gesture1=pred_gesture(pred1,0)
        gesture2=pred_gesture(pred2,0)
        pred_array1.append(str(gesture1[0]))
        pred_array2.append(str(gesture2[0]))
        
        mean_distro=fuse_mean(distro1,distro2)
        gesture_mean=pred_from_distro(classlabels,mean_distro)
        pred_arrayfusion.append(str(gesture_mean))
        
        #conf_distro=fuse_conf(distro1,distro2)
        #gesture_conf=pred_from_distro(classlabels,conf_distro)
        #pred_arrayfusion.append(str(gesture_conf))
        
        print(time.time(),'\n','Mode 1: ',gesture1,'\n','Mode 2: ',gesture2,'\n','Fusion: ',gesture_mean,'\n-------------')
       
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
        scores = eval_fusion(pred_array1,pred_array2,pred_arrayfusion,state)
        print(scores)
        acc = pd.DataFrame([scores],columns=['Mode 1','Mode 2','Fusion'])
        result = pd.DataFrame(list(zip(pred_array1,pred_array2,pred_arrayfusion)),columns=['Mode 1','Mode 2','Fusion'])
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
    
    onlinemode=0#int(input('Classifying live? 1:Y 0:N '))
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
        liveclassify()
