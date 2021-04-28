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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename

def eval_acc(results,state):
    count=results.count(state)
    acc=count/len(results)
    return acc

def eval_fusion(mode1,mode2,fusion,state):
    acc1=eval_acc(mode1,state)
    acc2=eval_acc(mode2,state)
    accF=eval_acc(fusion,state)
    return [acc1,acc2,accF]

def load_model(name,path):
    title='select saved model for '+name
    Tk().withdraw()
    model_name=askopenfilename(initialdir=path,title=title,filetypes = (("sav files","*.sav"),("all files","*.*")))
    with open(model_name,'rb') as load_model:
        model = pickle.load(load_model)
    return model

def matrix_from_csv_file(file):
    csv_data=pd.read_csv(file,delimiter=",").values
    matrix = csv_data[1:]
    headers = csv_data[0]
    print ('MAT', (matrix.shape))
	#print ('HDR', (headers.shape))
    return matrix, headers

def train_offline():
    path=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    title='select training dataset'
    Tk().withdraw()
    train_set=askopenfilename(initialdir=path,title=title,filetypes = (("csv files","*.csv"),("all files","*.*")))
    title_sav='location for trained model'
    modeldest=askdirectory(initialdir=path,title=title_sav)
    modeltype='gaussNB'
    modelname=modeldest+'/'+os.path.basename(train_set)[:-4]+'_'+modeltype+'.sav'
    if modeltype=='gaussNB':
        train_nb(train_set,modelname)

def train_nb(train_dat,model_path):
    data=matrix_from_csv_file(train_dat)[0]
    naivb=GaussianNB()
    naivb,acc=train_model(naivb,data)
    print('accuracy: ',acc)
    with open(model_path,'wb') as savepath:
        pickle.dump(naivb,savepath)
    return

def train_model(model,data):
    train=data[:,:-1]
    targets=data[:,-1]
    train1,train2,test1=np.array_split(train,3)
    train1=train1.astype(np.float64)
    train2=train2.astype(np.float64)
    traindat=np.concatenate((train1,train2))
    test1=test1.astype(np.float64)
    targets1,targets2,targetstest=np.array_split(targets,3)
    targetsdat=np.concatenate((targets1,targets2))
    model.fit(traindat,targetsdat)
    results=model.predict(test1)
    acc=accuracy_score(targetstest,results)
    return model, acc

def classify_instance(frame,model):
    prediction=frame
    return prediction

def classify_continuous(data):
    while True:
        pred=classify_instance(data,'NB')
        yield pred