#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 22:37:59 2021

@author: pritcham

module to contain functionality related to ML classification
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import sklearn as skl

def classify_instance(frame,model):
    prediction=frame
    return prediction

def classify_continuous(data):
    while True:
        pred=classify_instance(data,'NB')
        yield pred