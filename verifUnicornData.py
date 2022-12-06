#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 17:01:14 2022

@author: michael
"""

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
import handleBFSigProc as handleBF

from tkinter import Tk
from tkinter.filedialog import askopenfilename

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, NoiseTypes, DetrendOperations, WindowFunctions #WindowOperations in BF_v5
from brainflow.exit_codes import BrainflowExitCodes #,BrainFlowError
from brainflow.utils import check_memory_layout_row_major

test_datafile='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EEG/Raw/0001a-grasp-2-_EEG.csv'
trialname=test_datafile.split('/')[-1][:-9]
print(trialname)

def plot_signals(data,channels_to_plot,title='time domain',transposed=False):
    #df = pd.DataFrame(np.transpose(data))
    if transposed:
        data=data.transpose()
    df = pd.DataFrame(data)
    plt.figure()
    #plt.axis([0,18,-100,1000])
    #plt.ylim([188000, 190000])
    df[channels_to_plot].plot(subplots=True)
    plt.title('time series signal '+title)
    plt.show()
    
def plot_psd(psd,title):
    plt.figure()
    plt.plot(psd[1],psd[0])
    plt.title('power spectral density '+title)
    plt.show()

def check_PSD(data,channel,nfft,sampling_rate,title):
    psd=DataFilter.get_psd_welch(data[eeg_channel], nfft, nfft // 2, sampling_rate,
                               WindowFunctions.BLACKMAN_HARRIS.value)
    plot_psd(psd,title)
    band_power_alpha = DataFilter.get_band_power(psd, 7.0, 13.0)
    band_power_beta = DataFilter.get_band_power(psd, 14.0, 30.0)
    print("alpha/beta:%f", band_power_alpha / band_power_beta)
    return psd

board_id = BoardIds.UNICORN_BOARD.value
params = BrainFlowInputParams()
board=BoardShim(board_id,params)
#print(board.get_version()) #current ver is 4.2, get_version added in 5
#board_descr = BoardShim.get_board_descr(board_id)
#sampling_rate = int(board_descr['sampling_rate'])
sampling_rate = BoardShim.get_sampling_rate(board_id)
nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
#eeg_channels = board_descr['eeg_channels']
eeg_channels = [channel for channel in BoardShim.get_eeg_channels(board_id)]
timestamp_channel = BoardShim.get_timestamp_channel(board_id)
eeg_channel=eeg_channels[1]


data,_=handleBF.load_raw_brainflow(datafile=test_datafile)
try:
    check_memory_layout_row_major(data[eeg_channel],1)
except Exception as e:
    if e.exit_code==BrainflowExitCodes.INVALID_ARGUMENTS_ERROR.value:
        data=data.transpose()
        check_memory_layout_row_major(data[eeg_channel],1)
        
data=data[:,250:] #'''WHY IS THIS NEEDED, WHAT CAUSES IMPULSE. HARDWARE?'''

plot_signals(data,eeg_channel,title='before anything',transposed=True)
check_PSD(data,eeg_channel,nfft,sampling_rate,title='before anything')


'''DETREND THE DATA'''
#DC blocking filter may be able to realtime detrend
#https://dsp.stackexchange.com/questions/25189/detrending-in-real-time
DataFilter.detrend(data[eeg_channel], DetrendOperations.CONSTANT.value)

plot_signals(data,eeg_channel,title='after detrend',transposed=True)
check_PSD(data,eeg_channel,nfft,sampling_rate,title='after detrend')


'''50 Hz MAINS NOTCH'''
DataFilter.perform_bandstop(data[eeg_channel], sampling_rate, 48.0, 52.0, 3,
                                        FilterTypes.BUTTERWORTH.value, 0)

plot_signals(data,eeg_channel,title='after mains notch',transposed=True)
check_PSD(data,eeg_channel,nfft,sampling_rate,title='after mains notch')


'''HIGH PASS'''
DataFilter.perform_highpass(data[eeg_channel], sampling_rate, 3.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0) #0 is chebyshev ripple

plot_signals(data,eeg_channel,title='after 2.0Hz HPF',transposed=True)
check_PSD(data,eeg_channel,nfft,sampling_rate,title='after 2.0Hz HPF')


'''LOW PASS'''
DataFilter.perform_lowpass(data[eeg_channel], sampling_rate, 90.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0) #0 is chebyshev ripple

plot_signals(data,eeg_channel,title='after 90.0Hz LPF',transposed=True)
check_PSD(data,eeg_channel,nfft,sampling_rate,title='after 90.0Hz LPF')



