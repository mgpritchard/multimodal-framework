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
#import plotly as pltly

from tkinter import Tk
from tkinter.filedialog import askopenfilename

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, NoiseTypes

def matrix_from_csv_file(file_path,delimiter=','):
    csv_data = np.genfromtxt(file_path, delimiter=delimiter,dtype='float64')
    full_matrix = csv_data[1:]
    #headers = csv_data[0] # Commented since not used or returned [fcampelo]
    return full_matrix

def plot_eeg(data,eeg_channels,title='time domain'):
    #df = pd.DataFrame(np.transpose(data))
    df = pd.DataFrame(data)
    plt.figure()
    plt.title(title)
    #plt.axis([0,18,-100,1000])
    #plt.ylim([188000, 190000])
    df[eeg_channels].plot(subplots=True)
    plt.show()
    
def highpass(data,channels,samplerate,cutoff,order,filtertype,ripple=0):
    data=np.array(np.transpose(data),order='c')
    for count, channel in enumerate(channels):
        #thischannel = np.array(data[:,channel],order='C')
        DataFilter.perform_highpass(data[channel], samplerate, cutoff, order, filtertype, ripple)
    data=np.transpose(data)
    return data
    
def notch(data,channels,samplerate,notchfreq,bandwith,order,filtertype,ripple=0):
    data=np.array(np.transpose(data),order='c')
    for count, channel in enumerate(channels):
        DataFilter.perform_bandstop(data[channel], samplerate, notchfreq, bandwith, order, filtertype, ripple)
    data=np.transpose(data)
    return data

def plot_fft(data,samplefreq,title='FFT'):
    Y    = np.fft.fft(data[:,1])
    freq = np.fft.fftfreq(len(data[:,1]),1/samplefreq)
    plt.figure()
    #plt.axis([-5, 150, 0, 10000000]) #x_min,x_max,y_min,y_max
    plt.xlim([-5, 150])
    plt.ylim([-1e3,1e8])
    plt.title(title)
    plt.plot(freq,np.abs(Y))
    #plt.plot(freq,Y.real)
    plt.show()
    
def ref_to_avg(data):
    print('\nShould this be doing Surface Laplace instead?\n')
    for rowcount, row in enumerate(data):
        newrow=np.zeros(8)
        for colcount, col in enumerate(row):
            newrow[colcount] = col - np.mean(row)
        data[rowcount]=newrow
    return data

def load_raw_brainflow(datafile=None,):
    #board_id = BoardIds.PLAYBACK_FILE_BOARD.value #this lets you try signal proc algs in realtime without a device
    board_id = BoardIds.UNICORN_BOARD.value
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    #NEED ONCE TIME COL IS MOVED: eeg_channels = [channel+1 for channel in BoardShim.get_eeg_channels(board_id)]
    eeg_channels = [channel for channel in BoardShim.get_eeg_channels(board_id)]
    timestamp_channel = 17
    needed_channels=eeg_channels.copy()
    needed_channels.insert(0,timestamp_channel)

    if datafile is None:
        homepath=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        Tk().withdraw()
        datafile=askopenfilename(title='eeg file',initialdir=homepath)
    data=matrix_from_csv_file(datafile,delimiter='\t')
    
    data=data[:,needed_channels]
    
    return data, sampling_rate

def test_eeg_filt_pipeline(datafile=None,):
    
    #board_id = BoardIds.PLAYBACK_FILE_BOARD.value #this lets you try signal proc algs in realtime without a device
    board_id = BoardIds.UNICORN_BOARD.value
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    #NEED ONCE TIME COL IS MOVED: eeg_channels = [channel+1 for channel in BoardShim.get_eeg_channels(board_id)]
    eeg_channels = [channel for channel in BoardShim.get_eeg_channels(board_id)]
    #print(eeg_channels)

    if datafile is None:
        homepath=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        Tk().withdraw()
        datafile=askopenfilename(title='eeg file',initialdir=homepath)
    data=matrix_from_csv_file(datafile,delimiter='\t')
    
    #plot_eeg(data,eeg_channels)
    plot_eeg(data,1,'raw')
    data[:,eeg_channels]=ref_to_avg(data[:,eeg_channels])
    plot_eeg(data,1,'raw after ref') #col 1 offset goes to -100k


    plot_fft(data[:,eeg_channels],sampling_rate,'FFT pre-filt')
    hp_filt_type=FilterTypes.BUTTERWORTH.value
    hp_cutoff=0.5
    hp_order=3
    data=highpass(data,eeg_channels,sampling_rate,hp_cutoff,hp_order,hp_filt_type)
    plot_eeg(data,1,'HPF') #col 1 offset goes to -1000
    plot_fft(data[:,eeg_channels],sampling_rate,'FFT HPF')

    #hp order 3 and notch order 3 does very good hpf but with big ripple

    notch_filt_type=FilterTypes.BUTTERWORTH.value
    notch_freq=50
    notch_width=1
    notch_order=3
    data=notch(data,eeg_channels,sampling_rate,notch_freq,notch_width,notch_order,notch_filt_type)
    plot_eeg(data,2,'Notch')
    plot_fft(data[:,eeg_channels],sampling_rate,'FFT Notch')
    
    return data

if __name__ == '__main__':
    #test_eeg_filt_pipeline()
    load_raw_brainflow()
