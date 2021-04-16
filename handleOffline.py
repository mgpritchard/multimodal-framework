#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 21:33:38 2021

@author: pricham

module to contain functionality related to offline operation & testing
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames

def matrix_from_csv_file(file):
    csv_data=pd.read_csv(file,delimiter=",").values
    matrix = csv_data[1:]
    headers = csv_data[0]
    print ('MAT', (matrix.shape))
	#print ('HDR', (headers.shape))
    return matrix, headers

def loadset(label):
    path=os.path.realpath(__file__)
    title='select time series data for '+label
    Tk().withdraw()
    datafile=askopenfilename(initialdir=path,title=title,filetypes = (("csv files","*.csv"),("all files","*.*")))
    data=matrix_from_csv_file(datafile)[0]
    return data

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