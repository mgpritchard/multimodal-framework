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

def liveclassify():
    threadEMG=Thread(target=handleEMG.read_emg(),daemon=True)
    threadEEG=Thread(target=handleEMG.read_emg(),daemon=True)
    threadML=Thread(target=handleML.classify_continuous(),daemon=True)
    threadEMG.start()
    threadEEG.start()
    threadML.start()
    return

def offlineclassify():
    return

if __name__ == '__main__':
    onlinemode=int(input('Classifying live? 1:Y 0:N '))
    print(onlinemode)
    
    if not onlinemode:
        offlineclassify()
    else:
        liveclassify()
