#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 21:20:45 2021

@author: pritcham

module to contain functionality related to [decision-level] fusion
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

def fuse_conf(mode1,mode2):
    fused=[]
    for instance in range(len(mode1)):
        if max(mode1[instance,:])>max(mode2[instance,:]):
            fused.append(mode1)
        else:
            fused.append(mode2)
    return fused

def fuse_linweight(mode1,mode2,weight1,weight2):
    total_weight=weight1+weight2
    weighted_1=(mode1*weight1)/total_weight
    weighted_2=(mode2*weight2)/total_weight
    fused=weighted_1+weighted_2
    return fused

def fuse_mean(mode1,mode2):
    mean=fuse_linweight(mode1,mode2,50,50)
    return mean
    
#def adaptive_fusion:
 #   try_some
  #  pick_best_strat
    #report strat to top level and then call this again after a while??