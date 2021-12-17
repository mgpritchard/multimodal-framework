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
    
    