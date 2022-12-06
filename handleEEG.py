#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:16:08 2021

@author: michael

module to contain functionality related to EEG acquisition and processing
"""
#BOARD CREATED ERROR HAPPENS AFTER CRASH, JUST RESTART KERNEL

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from handleML import *
from handleOffline import get_feats
import time
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import argparse

# https://stackoverflow.com/questions/6663222/doing-fft-in-realtime
#successive fouriers, slide the window slightly
#in a realtime: slide the window by x ms everytime you get a new x ms data

#https://web.archive.org/web/20181105231756/http://developer.choosemuse.com/tools/available-data#Absolute_Band_Powers
#steps through the process incl sample rate to freq bins calc
#muse use Hamming window, 256 samples, to get 128 components +0Hz
#then basically sum up components within a band (allowing components to be in multiple bands)

#https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf
#PAGE 10 - 11, short answer is Hamming or Hanning would likely do the job.
# note comparison of the two on p12. Narrow main lobe preferred as spectral
#resolution and separation of signals is important (p11) so likely Hamming

#https://www.researchgate.net/publication/336192872_Window_Functions_Analysis_in_Filters_for_EEG_Movement_Intention_Signals
#suggests Hamming and Hann broadly similar (Bartlett best but not in brainflow)
#ADD THIS ONE TO LIT ONGOING


def main():
    BoardShim.enable_dev_board_logger()
    '''
    #https://brainflow.readthedocs.io/en/stable/SupportedBoards.html#unicorn
    parser = argparse.ArgumentParser()
    #parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    #parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    #ser num only if multiple?
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=True)
    args = parser.parse_args()
    
    #UNICORN: ID = 8
    #SYNTHETIC: ID = -1
    '''
    #board_id=-1
    board_id=8
    '''TEST THE UNICORN CONNECTION INDEPEDENTLY'''
    params = BrainFlowInputParams()
    #params.serial_number = args.serial_number
    #params.file = args.file #only for playback

    #board = BoardShim(args.board_id, params)
    board=BoardShim(board_id,params)
    board.prepare_session()

    board.start_stream () # use this for default options
    #board.start_stream(45000, args.streamer_params)
    time.sleep(10)  
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data = board.get_board_data()  # get all data and remove it from internal buffer
    board.stop_stream()
    board.release_session()

    print(data)

def setup_bf(mode="synth"):
    BoardShim.enable_dev_board_logger()
    if mode=="unicorn":
        board_id=8
    else:
        board_id=-1
    params = BrainFlowInputParams()
    #params.serial_number = args.serial_number
    #params.file = args.file #only for playback

    #board = BoardShim(args.board_id, params)
    board=BoardShim(board_id,params)
    board.prepare_session()
    return board

def kill_bf(mode="synth"):
    BoardShim.enable_dev_board_logger()
    if mode=="unicorn":
        board_id=8
    else:
        board_id=-1
    params = BrainFlowInputParams()
    board=BoardShim(board_id,params)
    board.release_session()

def save_EEG(dataEEG,path,gesture,pptid):
    filename=path+'/'+pptid+'-'+gesture.label+'-'+str(gesture.rep)+'-'+'_EEG.csv'
    #pd.DataFrame(np.transpose(dataEEG)).to_csv(filename,header=None,index=None)
    #https://brainflow.readthedocs.io/en/stable/Examples.html#python
    #suggested the following as a better way than pd to save data:
    DataFilter.write_file(dataEEG, filename, 'w')
    

if __name__ == "__main__":
    #main()
    duration=3
    board=setup_bf("unicorn")
    tstart=time.time()
    tend=tstart+duration
    board.start_stream()
    while time.time()<tend:
        #print('\b+',end=' ') #this might crash it lol
        pass
    data = board.get_board_data()  # get all data and remove it from internal buffer
    board.stop_stream()
    board.release_session()

    print('\n',data)
    
    