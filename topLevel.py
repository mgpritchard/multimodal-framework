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
from tkinter.filedialog import askopenfilename, askopenfilenames
from queue import Queue
import logging
from threading import Thread
from time import time
import handleEMG
import handleML
from handleOffline import *

def liveclassify():
    threadEMG=Thread(target=handleEMG.read_emg(),daemon=True)
    threadEEG=Thread(target=handleEMG.read_emg(),daemon=True)
    threadML=Thread(target=handleML.classify_continuous(),daemon=True)
    threadEMG.start()
    threadEEG.start()
    threadML.start()
    return

def offlineclassify():
    mode1=loadset('first mode')
    mode2=loadset('second mode')
    quickplot(mode1,'mode 1 before')
    quickplot(mode2,'mode 2 before')
    mode1,mode2=sync_crop(mode1,mode2)
    quickplot(mode1,'mode 1 after')
    quickplot(mode2,'mode 2 after')
    #do time windows. get feats from window 1 and send to classifier/fuser
    #if ___ms has not passed, wait until it has. then get next rolling window
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
    
    if not onlinemode:
        offlineclassify()
    else:
        liveclassify()
