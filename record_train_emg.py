#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 22:08:53 2021

@author: pritcham

functions related to recording and training of emg classifiers
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from tkinter import *
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory
from PIL import ImageTk,Image
from queue import Queue
import logging
from threading import Thread
import time
from handleEMG import *
#from handleEEG import *
from handleML import *
from handleOffline import *
from handleFusion import *
from handlePrompts import *
import pickle
import random
import brainflow


def record_gesture(path,gesture,duration,pptid,rep):
    pyoc_record_fixed_time(path, duration)
    rename_latest_emg(path, pptid, gesture.label, gesture.rep)

def show_and_record(gesture,pptid,path,duration,figwin,gestlist,count):
    time.sleep(1)
    display_prompt(figwin,gesture,gestlist,count)
    pyoc_record_fixed_time(path, duration) #emg recording has to start after
            #prompt as it is a blocking method; thread not released until end
    rename_latest_emg(path,pptid,gesture.label,gesture.rep)
    gesture.rep+=1
    
    #would be good to rewrite such that recording carries on for x seconds
    #following the rest prompt, with another marker... but again emg recording
    #is a blocking method. only option may be to spin up another thread?


def record_exptl(pptid,path,duration,numreps,resttime):
    gestlist=[]
    gests=setup_default_gests()
    rest = Gesture("rest","/home/michael/Documents/Aston/MultimodalFW/prompts/space.jpg")
    for gest in gests:
        gestlist.extend(gest for i in range(numreps))
    random.shuffle(gestlist)
    figwin=display_setup(gestlist)
    count=0
    for gesture in gestlist:
        count+=1
        show_and_record(gesture,pptid,path,duration,figwin,gestlist,count)
        display_prompt(figwin,rest,gestlist,count)
        time.sleep(resttime)
    figwin.destroy()
    print('All done! Thanks for your contribution')
    

def quickplot(data,label):
    fig,ax=plt.subplots()
    ax.plot(data[:,0],data[:,1])
    ax.set(title=label)
    plt.show()

if __name__ == '__main__':
    
    recording=1
    genset=0
    train=0
    if recording:
    
        pptid=input('particpant id: ')
        startpath=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        #myo_connect_setup()
        #record_exptl(pptid)
        Tk().withdraw()
        path=askdirectory(initialdir=startpath,title='where to record EMG?')
        duration=5
        numreps=5
        resttime=1  #randomised 10-12
        #gesture='testing'
    
        record_exptl(pptid,path,duration,numreps,resttime)
    elif genset:
        gen_train_set()
    elif train:
        train_offline('LDA')