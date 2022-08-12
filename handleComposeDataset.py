#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 19:28:17 2021

@author: pritcham
"""

import os
from tkinter import *
#from feature_extraction.EMG_generate_training_matrix import gen_training_matrix
#from feature_extraction.labelstoClass import numtoClass
from distutils.dir_util import copy_tree
from shutil import copyfile
#from distutils.dir_util import copyfile

import platform
import subprocess

def open_file(path): #unused
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

def print_if_sel(i):
    if ppts[i].get():
        print(pptlist[i])

def listppts():
    print('Selected participants:')
    for i in range(9):
        print_if_sel(i)
        
def build_path(path,num):
    while len(num)<3:
        num='0'+num
    path=path+num
    return path

def build_set(paths,destination):
    for path in paths:
        #pathname=os.path.basename(path)
        for file in os.listdir(path):
            dest=os.path.join(destination,file)
            source=os.path.join(path,file)
            copyfile(source,dest)
        #print(path)
        #print(dest)
        #copyfile(paths(path),dest)
        #copy_tree(path,dest)
    #open_file(destination)
    
def build_set_separate_modes(paths,destination):
    emgdest=destination+'EMG/'
    eegdest=destination+'EEG/'
    if not os.path.exists(emgdest):
        print(emgdest)
        os.makedirs(emgdest)
    if not os.path.exists(eegdest):
        print(eegdest)
        os.makedirs(eegdest)
    for path in paths:
        for file in os.listdir(path):
            if file[-7:-4]=='EEG':
                dest=os.path.join(eegdest,file)
            else:
                dest=os.path.join(emgdest,file)
            source=os.path.join(path,file)
            copyfile(source,dest)

if __name__ == '__main__':
    root=Tk()
    root.geometry('175x375')
    devset = ['1 - M 24','2 - M 42','4 - F 36',
               '7 - F 29','8 - F 27','9 - F 24',
               '11 - M 24','13 - M 31','14 - M 28']
    holdout = ['3 - M 36','5 - M 20','6 - M 29','10 - F 24','12 - F 26']
    whichset='dev'
    if whichset=='dev':
        pptlist=devset
    elif whichset=='holdout':
        pptlist=holdout
    ppts = []
    for i in range(len(pptlist)):
        ppts.append(IntVar())
        Checkbutton(root,text=pptlist[i],variable=ppts[i]).grid(row=i,sticky=W,pady=3)
    #Button(root,text='Quit',command=root.quit).grid(row=3, sticky=W, pady=4)
    width,height=root.grid_size();
    Button(root,text='Check',command=listppts).grid(row=height,sticky=W)
    Button(root,text='Selected',command=root.destroy).grid(row=height,column=width,sticky=E)
    '''#above may not be working, may need to close pupup directly?'''
    root.mainloop()
    path_base='/home/michael/Documents/Aston/MultimodalFW/'
    path_holdoutset=path_base+'dataset/holdout/'
    path_devset=path_base+'dataset/dev/'
    #trypath=build_path(7)
    #print(trypath)
    paths=[]
    for i in range(len(pptlist)):
        if ppts[i].get():
            print('using '+str(pptlist[i]))
            paths.append(build_path(path_devset,pptlist[i].split(' ')[0]))
    #print(paths)
    destinationroot="/home/michael/Documents/Aston/MultimodalFW/working_dataset/"
    destination=destinationroot + whichset + '/'
    if not os.path.exists(destination):
        print(destination)
        os.makedirs(destination)
    #build_set(paths,destination)
    build_set_separate_modes(paths,destination)
    
    
    
    
    '''matrix_path=destination + "_feats.csv"
    print(destination)
    ranking_path=destination + "_ranking.csv"
    gen_set=0
    if gen_set:
        gen_training_matrix(destination,matrix_path,-1)
        gen_training_matrix(destination,ranking_path,-1,enblRanking=1)
        labelled_path=destination+"_featsClass.csv"
        labelled_ranking=destination+"_rankClass.csv"
        numtoClass(matrix_path,labelled_path)
        numtoClass(ranking_path,labelled_ranking)'''
    
