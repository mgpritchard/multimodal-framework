#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 19:37:04 2022

@author: pritcham

module for processing eeg with brainflow
"""
#https://brainflow.readthedocs.io/en/stable/Examples.html#python-signal-filtering
#https://brainflow.readthedocs.io/en/stable/UserAPI.html#brainflow-data-filter


import os
import argparse
import time
import brainflow
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tkinter import Tk
from tkinter.filedialog import askopenfilename

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, NoiseTypes

def matrix_from_csv_file(file_path):
    csv_data = np.genfromtxt(file_path, delimiter = ',')
    full_matrix = csv_data[1:]
    #headers = csv_data[0] # Commented since not used or returned [fcampelo]
    return full_matrix

def plot_eeg(data,eeg_channels):
    df = pd.DataFrame(np.transpose(data))
    plt.figure()
    df[eeg_channels].plot(subplots=True)
    plt.show()
    
def highpass(data,channels,samplerate,cutoff,order,filtertype,ripple=0):
    for count, channel in enumerate(channels):
        DataFilter.perform_highpass(data[channel], samplerate, cutoff, order, filtertype, ripple)
    
def notch(data,channels,samplerate,notchfreq,bandwith,order,filtertype,ripple=0):
    for count, channel in enumerate(channels):
        DataFilter.perform_bandstop(data[channel], samplerate, notchfreq, bandwith, order, filtertype)

def eeg_filt_pipeline(datafile=None):
    
    #board_id = BoardIds.PLAYBACK_FILE_BOARD.value #this lets you try signal proc algs in realtime without a device
    board_id = BoardIds.UNICORN_BOARD.value
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eeg_channels = [channel+1 for channel in BoardShim.get_eeg_channels(board_id)]
    #print(eeg_channels)
    filter_type=FilterTypes.BUTTERWORTH.value

    if datafile is None:
        homepath=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        Tk().withdraw()
        datafile=askopenfilename(title='eeg file',initialdir=homepath)
    data=matrix_from_csv_file(datafile)
    plot_eeg(data,eeg_channels)

    hp_cutoff=0.5
    hp_order=3
    highpass(data,eeg_channels,sampling_rate,hp_cutoff,hp_order,filter_type)
    plot_eeg(data,eeg_channels)

    notch_freq=50
    notch_width=1
    notch_order=3
    notch(data,eeg_channels,sampling_rate,notch_freq,notch_width,notch_order,filter_type)
    plot_eeg(data,eeg_channels)
    
    return data