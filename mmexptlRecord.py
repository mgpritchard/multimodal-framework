#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 22:08:53 2021

@author: pritcham

script for recording gestures as part of multimodal experiment
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
from handleEEG import *
from handleML import *
from handleOffline import *
from handleFusion import *
from handlePrompts import *
import pickle
import random
import brainflow
import traceback
import params

def record_gesture(path,gesture,duration,pptid,rep):
    pyoc_record_fixed_time(path, duration)
    rename_latest_emg(path, pptid, gesture.label, gesture.rep)

def show_and_record(gesture,pptid,path,duration,figwin,gestlist,count,boardEEG,m):
    boardEEG.start_stream()
    time.sleep(1) #ensure eeg is recording before prompt is shown
    display_prompt(figwin,gesture,gestlist,count)
    boardEEG.insert_marker(1) #marker in eegstream to denote prompt being shown
    #pyoc_record_fixed_time(path, duration) #emg recording has to start after
            #prompt as it is a blocking method; thread not released until end
    pyoc_record_no_init(path, duration, m)
    rename_latest_emg(path,pptid,gesture.label,gesture.rep)
    dataEEG = boardEEG.get_board_data()  # get all data and remove it from internal buffer
    boardEEG.stop_stream()
    save_EEG(dataEEG,path,gesture,pptid)
    gesture.rep+=1
    
    #would be good to rewrite such that recording carries on for x seconds
    #following the rest prompt, with another marker... but again emg recording
    #is a blocking method. only option may be to spin up another thread?


def record_exptl(pptid,path,duration,numreps,resttime):
    try:
        boardEEG=setup_bf()
    except Exception:
        kill_bf()
        boardEEG=setup_bf()
        
    try:
        gestlist=[]
        gests=setup_default_gests()
        rest = Gesture("rest",params.prompt_neut)
        for gest in gests:
            gestlist.extend(gest for i in range(numreps))
        random.shuffle(gestlist)
        figwin=display_setup(gestlist)
        count=0
        m=pyoc_init()
        for gesture in gestlist:
            count+=1
            duration=random.randint(400,500)/100
            resttime=random.randint(1000,1200)/100
            show_and_record(gesture,pptid,path,duration,figwin,gestlist,count,boardEEG,m)
            '''tstart=time.time()      #swap out with above if testing timing
            tend=tstart+duration
            print('start: ',tstart,'\n end: ',tend)
            time.sleep(duration)'''
            display_prompt(figwin,rest,gestlist,count)
            print('rest now')
            time.sleep(resttime)
            print('rested')
    except Exception as e:
        boardEEG.release_session()
        figwin.destroy()
        print(e)
        print(traceback.format_exc())
    else:
        figwin.destroy()
        boardEEG.release_session()  #NEEDS TO REACH TO AVOID KERNEL RESTART!
        print('All done! Thanks for your contribution')
    

def quickplot(data,label):
    fig,ax=plt.subplots()
    ax.plot(data[:,0],data[:,1])
    ax.set(title=label)
    plt.show()

if __name__ == '__main__':
    
    pptid=input('particpant id: ')
    startpath=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    #myo_connect_setup()
    #record_exptl(pptid)
    '''
    Tk().withdraw()
    path=askdirectory(initialdir=startpath,title='where to dump EMG?')
    '''
    #path='C:\AstonApps\Solidworks2019SP3'
    path=params.path
    duration=5
    numreps=1
    resttime=1  #randomised 10-12
    gesture='testing'

    record_exptl(pptid,path,duration,numreps,resttime)

    #for i in range(3):
     #   record_gesture(path, gesture, duration, pptid, (i+1))
    #pyoc_record_fixed_time(path, 15)
    #rename_latest_emg(path, pptid, 'testing', 1)
    #pyoc_record_fixed_time(path, 15)
    #rename_latest_emg(path, pptid, 'testing', 2)
