#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 22:12:09 2021

@author: pritcham

module to contain functionality related to EMG acquisition and processing
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from myo_live import *
from handleML import *
from handleOffline import get_feats
from handleFusion import *
from handlePrompts import *
import time

def pyoc_fuse_emgonly(model_mode1,model_mode2,method='dupe',fusion=True,outp='print',limit=0):

    m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)

    def proc_emg(emg, moving, times=[]):
        ## print framerate of received data
        times.append(time.time())
        if len(times) > 20:
            #print((len(times) - 1) / (times[-1] - times[0]))
            times.pop(0)
                 
    global currentemg
    currentemg=np.zeros((1,8),dtype=int)
    global emgarr
    emgarr=np.zeros((1,8),dtype=int)
    
    if outp=='window':
        predwin=display_predictions_setup()
    
    m.add_emg_handler(proc_emg)
    m.connect()

    period=1
    loud=0
    classlabels=model_mode1.classes_
    mode1arr=[]
    mode2arr=[]
    fusionarr=[]
    try:
        tstart=time.time()
        setUpDone=False
        while np.size(fusionarr,0)<=limit if limit else True:
            currentemg=m.run_report(1)
            if currentemg is not None:
                if loud:
                    print(currentemg)
                temparr=np.array(list(currentemg))
            else:
                continue
            emgarr=np.concatenate((emgarr,temparr),axis=0)

            if time.time()-tstart >= (period/2):
                if loud:
                    print('start: ',tstart,'\nend: ',time.time(),'\nperiod: ',period)                                                         
                #values=get_feats(emgarr)
                if method=='dupe':
                    values_1=get_feats(emgarr)
                    values_2=get_feats(emgarr)
                elif method=='split':
                    values_1=get_feats(emgarr[:,:4])
                    values_2=get_feats(emgarr[:,4:])
                emgarr=[emgarr[-(int(len(emgarr)/2))]]
                
                distro1= prob_dist(model_mode1,values_1)
                distro2= prob_dist(model_mode2,values_2)
                mode1arr.append(distro1)
                mode2arr.append(distro2)
                gesture1=pred_from_distro(classlabels, distro1)
                gesture2=pred_from_distro(classlabels, distro2)
                
                if outp=='print':
                    print(time.time(),'\n','Mode 1: ',gesture1,'\n','Mode 2: ',gesture2)
                 
                if fusion:
                    if setUpDone==False:
                        mean_distro=fuse_mean(distro1,distro2)
                        fusionarr.append(mean_distro)
                        gesture_mean=pred_from_distro(classlabels,mean_distro)
                        if outp=='window':
                            display_preds_and_weights(predwin,gesture1,gesture2,gesture_mean,0.5,0.5)
                        else:
                            print(time.time(),'\n','Mean: ',gesture_mean)
                            print('-------------')
                       
                        if len(mode1arr)>=10:
                            w1,w2=get_weights(np.stack(mode1arr,axis=1)[0], np.stack(mode2arr,axis=1)[0],method='autocorr')#'JS'
                            fuse_distro=fusionarr[-1]
                            setUpDone=True
                            
                            
                    elif setUpDone==True:
                        fuse_distro=fuse_single(distro1,distro2,w1,w2,fuse_distro,loud)
                        fusionarr.append(fuse_distro)
                        gesture_fuse=pred_from_distro(classlabels,fuse_distro)
                        if outp=='window':
                            display_preds_and_weights(predwin,gesture1,gesture2,gesture_fuse,w1,w2)
                        else:
                            print(time.time(),'\n','Fusion: ',gesture_fuse)
                            print('-------------')
                        
                        if len(mode1arr)%10==0:
                            w1,w2=get_weights(np.stack(mode1arr,axis=1)[0], np.stack(mode2arr,axis=1)[0],method='autocorr')#'JS'
                            fuse_distro=fusionarr[-1]
                        
                tstart=time.time()
        m.disconnect()
        predwin.destroy()
        return mode1arr,mode2arr,fusionarr
         
    except KeyboardInterrupt:
        m.disconnect()
        predwin().destroy()
        return mode1arr,mode2arr,fusionarr
        pass
    finally:
        print()
        
        
def pyoc_fuse_adapt_autocorr(model_mode1,model_mode2,method='dupe',fusion=True,outp='print',limit=0,autocorr_lag=3):

    m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)

    def proc_emg(emg, moving, times=[]):
        ## print framerate of received data
        times.append(time.time())
        if len(times) > 20:
            #print((len(times) - 1) / (times[-1] - times[0]))
            times.pop(0)
                 
    global currentemg
    currentemg=np.zeros((1,8),dtype=int)
    global emgarr
    emgarr=np.zeros((1,8),dtype=int)
    
    if outp=='window':
        predwin=display_predictions_setup()
    
    m.add_emg_handler(proc_emg)
    m.connect()

    period=1
    loud=0
    classlabels=model_mode1.classes_
    mode1arr=[]
    mode2arr=[]
    fusionarr=[]
    try:
        tstart=time.time()
        setUpDone=False
        while np.size(fusionarr,0)<=limit if limit else True:
            currentemg=m.run_report(1)
            if currentemg is not None:
                if loud:
                    print(currentemg)
                temparr=np.array(list(currentemg))
            else:
                continue
            emgarr=np.concatenate((emgarr,temparr),axis=0)

            if time.time()-tstart >= (period/2):
                if loud:
                    print('start: ',tstart,'\nend: ',time.time(),'\nperiod: ',period)                                                         
                #values=get_feats(emgarr)
                if method=='dupe':
                    values_1=get_feats(emgarr)
                    values_2=get_feats(emgarr)
                elif method=='split':
                    values_1=get_feats(emgarr[:,:4])
                    values_2=get_feats(emgarr[:,4:])
                emgarr=[emgarr[-(int(len(emgarr)/2))]]
                
                distro1= prob_dist(model_mode1,values_1)
                distro2= prob_dist(model_mode2,values_2)
                mode1arr.append(distro1)
                mode2arr.append(distro2)
                gesture1=pred_from_distro(classlabels, distro1)
                gesture2=pred_from_distro(classlabels, distro2)
                
                if outp=='print':
                    print(time.time(),'\n','Mode 1: ',gesture1,'\n','Mode 2: ',gesture2)
                 
                if fusion:
                    if setUpDone==False:
                        mean_distro=fuse_mean(distro1,distro2)
                        fusionarr.append(mean_distro)
                        gesture_mean=pred_from_distro(classlabels,mean_distro)
                        if outp=='window':
                            display_preds_and_weights(predwin,gesture1,gesture2,gesture_mean,0.5,0.5)
                        else:
                            print(time.time(),'\n','Mean: ',gesture_mean)
                            print('-------------')
                       
                        if len(mode1arr)>=autocorr_lag:
                            #w1,w2=get_weights(np.stack(mode1arr,axis=1)[0], np.stack(mode2arr,axis=1)[0],method='autocorr')
                            #fuse_distro=fusionarr[-1]
                            w1,w2=fuse_single_autocorr(distro1,distro2,fusionarr[(len(fusionarr)-autocorr_lag):],autocorr_lag)
                            setUpDone=True
                            
                            
                    elif setUpDone==True:
                        fuse_distro=fuse_linweight(distro1,distro2,w1,w2)
                        fusionarr.append(fuse_distro)
                        gesture_fuse=pred_from_distro(classlabels,fuse_distro)
                        if outp=='window':
                            display_preds_and_weights(predwin,gesture1,gesture2,gesture_fuse,w1,w2)
                        else:
                            print(time.time(),'\n','Fusion: ',gesture_fuse)
                            print('-------------')
                        
                        if len(mode1arr)%autocorr_lag==0:
                            w1,w2=fuse_single_autocorr(distro1,distro2,fusionarr[(len(fusionarr)-autocorr_lag):],autocorr_lag)
                         #   w1,w2=get_weights(np.stack(mode1arr,axis=1)[0], np.stack(mode2arr,axis=1)[0],method='autocorr')
                          #  fuse_distro=fusionarr[-1]
                        
                tstart=time.time()
        m.disconnect()
        predwin.destroy()
        return mode1arr,mode2arr,fusionarr
         
    except KeyboardInterrupt:
        m.disconnect()
        predwin().destroy()
        return mode1arr,mode2arr,fusionarr
        pass
    finally:
        print()
        
def pyoc_run(model):
    m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)

    def proc_emg(emg, moving, times=[]):
        ## print framerate of received data
        times.append(time.time())
        if len(times) > 20:
            #print((len(times) - 1) / (times[-1] - times[0]))
            times.pop(0)
                 
    global currentemg
    currentemg=np.zeros((1,8),dtype=int)
    global emgarr
    emgarr=np.zeros((1,8),dtype=int)
    
    m.add_emg_handler(proc_emg)
    m.connect()
    
    period=1
    loud=0
    try:
        tstart=time.time()
        while True:
            currentemg=m.run_report(1)
            if currentemg is not None:
                if loud:
                    print(currentemg)
                temparr=np.array(list(currentemg))
            else:
                continue
            emgarr=np.concatenate((emgarr,temparr),axis=0)

            #if len(emgarr)==150:
            if time.time()-tstart >= (period/2):
                print('start: ',tstart,'\nend: ',time.time(),'\nperiod: ',period)                                                         
 
                values=get_feats(emgarr)
                emgarr=[emgarr[-(int(len(emgarr)/2))]]
                
                pred = predict_from_array(model,values)
                distro = prob_dist(model,values)
                gesture=pred_gesture(pred,0)
                
                print(time.time(),'\n','Prediction: ',gesture)
                print('-------------')
        
                tstart=time.time()

    except KeyboardInterrupt:
        m.disconnect()
        pass
    finally:
        print()

def pyo_slice_and_class(model,tstart,period,emgarr):
    print('start: ',tstart,'\nend: ',time.time(),'\nperiod: ',period)                                                         
    #values=get_feats(emgarr)
    values=get_feats(emgarr[:,:4]) #testing with half model
    emgarr=[emgarr[-(int(len(emgarr)/2))]]
    
    pred = predict_from_array(model,values)
    distro = prob_dist(model,values)
    gesture=pred_gesture(pred,0)
    
    print(time.time(),'\n','Prediction: ',gesture)
    print('-------------')

    newtstart=time.time()
    
    return newtstart, gesture
        
def pyoc_setup(askparams):
    m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)

    def proc_emg(emg, moving, times=[]):
        times.append(time.time())
        if len(times) > 20:
            #print((len(times) - 1) / (times[-1] - times[0]))
            times.pop(0)
                 
    global currentemg
    currentemg=np.zeros((1,8),dtype=int)
    global emgarr
    emgarr=np.zeros((1,8),dtype=int)
    
    m.add_emg_handler(proc_emg)
    m.connect()
    
    period=1
    loud=0
    if askparams:
        period=input('period in seconds (default 1): ')
        loud=input('print incoming EMG? ')
    gesture_array=[]
    return m, period, loud, gesture_array, currentemg, emgarr

def pyoc_mainloop(model):
    m,period,loud,gesture_array,currentemg,emgarr = pyoc_setup(0)
    try:
        tstart=time.time()
        while True:
            currentemg=m.run_report(1)
            if currentemg is not None:
                if loud:
                    print(currentemg)
                temparr=np.array(list(currentemg))
            else:
                continue
            emgarr=np.concatenate((emgarr,temparr),axis=0)
            
            if time.time()-tstart >= (period/2):
                tstart,gesture=pyo_slice_and_class(model, tstart, period, emgarr)
                gesture_array.append(gesture)
                
    except KeyboardInterrupt:
        m.disconnect()
        pass
    finally:
        print()


def pyoc_oneslice(m,period,loud,gesture_array,currentemg,emgarr,model):
    tstart=time.time()
    while time.time()-tstart < (period/2):
        currentemg=m.run_report(1)
        if currentemg is not None:
            if loud:
                print(currentemg)
            temparr=np.array(list(currentemg))
        else:
            continue
        emgarr=np.concatenate((emgarr,temparr),axis=0) 
    print(emgarr)
    tstart,gesture=pyo_slice_and_class(model, tstart, period, emgarr)
    return gesture

def pyoc_slicetester(model):
    slice_nums=int(input('number of slices: '))
    m,period,loud,gesture_array,currentemg,emgarr = pyoc_setup(0)
    n=0
    gesture_array=[]
    tstart=time.time()
    while time.time()-tstart < (period/2):
        currentemg=m.run_report(1)
        if currentemg is not None:
            if loud:
                print(currentemg)
            temparr=np.array(list(currentemg))
        else:
            continue
        emgarr=np.concatenate((emgarr,temparr),axis=0)
    while n<slice_nums:
        gesture=pyoc_oneslice(m,period,loud,gesture_array,currentemg,emgarr,model)
        gesture_array.append(str(gesture[0]))
        n+=1
    print('done')
    m.disconnect()
    return gesture_array

def pyoc_create_instance():
    m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)
    
    
        
    global currentemg
    currentemg=np.zeros((1,8),dtype=int)
    global emgarr
    emgarr=np.zeros((1,8),dtype=int)
    
    
    m.connect(quiet=True)
    
    return m

def pyoc_record_fixed_from_existing(path,duration,m):
    destfile=(path+"/emg_data"+time.strftime("%Y%m%d-%H%M%S")+'.csv')
    with open(destfile,'w',newline='') as csvfile:
        emgwriter=csv.writer(csvfile, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        
        def proc_emg(emg, moving, emgwriter,times=[]):
            #print(emg)
            times.append(time.time())
            if len(times) > 20:
                #print((len(times) - 1) / (times[-1] - times[0]))
                times.pop(0)
            emgwrite=list(emg)
            emgwrite.insert(0,(int(round(time.time() * 1000))))
            emgwrite=tuple(emgwrite)
            emgwriter.writerow(emgwrite)
        m.add_emg_handler(proc_emg)
        
        tstart=time.time()
        tend=tstart+duration
        print('recording...')
        
        try:
            while True:
                m.run(1,emgwriter)
                if time.time()>=tend:
                    #print('now: ',time.time())
                    print('...recording finished')
                    raise KeyboardInterrupt()
    
        except KeyboardInterrupt:
            #m.disconnect()
            m.remove_emg_handler(proc_emg)
            pass
        finally:
            #m.disconnect()
            m.remove_emg_handler(proc_emg)
            print()

def pyoc_init():
    m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)
    global currentemg
    currentemg=np.zeros((1,8),dtype=int)
    global emgarr
    emgarr=np.zeros((1,8),dtype=int)
    return m

def pyoc_record_no_init(path,duration,m):
    destfile=(path+"/emg_data"+time.strftime("%Y%m%d-%H%M%S")+'.csv')
    with open(destfile,'w',newline='') as csvfile:
        emgwriter=csv.writer(csvfile, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        
        
        
        def proc_emg(emg, moving, times=[]):
            #print(emg)
            times.append(time.time())
            if len(times) > 20:
                #print((len(times) - 1) / (times[-1] - times[0]))
                times.pop(0)
            emgwrite=list(emg)
            emgwrite.insert(0,(int(round(time.time() * 1000))))
            emgwrite=tuple(emgwrite)
            emgwriter.writerow(emgwrite)
        
        #m.clear_emg_handlers()
        m.add_emg_handler(proc_emg)
        m.connect(quiet=True)
        tstart=time.time()
        tend=tstart+duration
        #print('start: ',tstart,'\n end: ',tend)
        #last_vals = None
        #timestamps=[]
        print('recording...')
    
        try:
            while True:
                m.run(1)
                if time.time()>=tend:
                    #print('now: ',time.time())
                    print('...recording finished')
                    raise KeyboardInterrupt()
    
        except KeyboardInterrupt:
            m.disconnect()
            m.remove_emg_handler(proc_emg)
            print('myo disconnected successfully')
            pass
        finally:
            m.disconnect()
            m.remove_emg_handler(proc_emg)
            print('myo disconnected successfully')
            print()

def pyoc_record_fixed_time(path,duration):
    destfile=(path+"/emg_data"+time.strftime("%Y%m%d-%H%M%S")+'.csv')
    with open(destfile,'w',newline='') as csvfile:
        emgwriter=csv.writer(csvfile, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        
        tstart=time.time()
        tend=tstart+duration
        #print('start: ',tstart,'\n end: ',tend)
        #last_vals = None
        #timestamps=[]
    
        m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)
    
        def proc_emg(emg, moving, times=[]):
            #print(emg)
            times.append(time.time())
            if len(times) > 20:
                #print((len(times) - 1) / (times[-1] - times[0]))
                times.pop(0)
            emgwrite=list(emg)
            emgwrite.insert(0,(int(round(time.time() * 1000))))
            emgwrite=tuple(emgwrite)
            emgwriter.writerow(emgwrite)

        global currentemg
        currentemg=np.zeros((1,8),dtype=int)
        global emgarr
        emgarr=np.zeros((1,8),dtype=int)
        
        m.add_emg_handler(proc_emg)
        m.connect(quiet=True)
        print('recording...')
    
        try:
            while True:
                m.run(1)
                if time.time()>=tend:
                    #print('now: ',time.time())
                    print('...recording finished')
                    raise KeyboardInterrupt()
    
        except KeyboardInterrupt:
            m.disconnect()
            print('myo disconnected successfully')
            pass
        finally:
            m.disconnect()
            print('myo disconnected successfully')
            print()

def rename_latest_emg(path,pptid,gesture,rep):
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.lower().startswith('emg_data'):
                os.rename(os.path.join(root, filename),os.path.join(root,(str(pptid)+'-'+gesture+'-'+str(rep)+'.csv')))
                
def delete_latest_emg(path):
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.lower().startswith('emg_data'):
                os.remove(os.path.join(root, filename))
'''-------------------------------------------------'''
def quick_train_splitemg():  #this but in real
    setup_myo
    
    print('************* \n Close Hand!')
    time.sleep(2)
    closedata=record_some_data
    print('data collected, rest your hand for a moment!')
    closefeats_h1=get_feats(closedata[:,:4]) . append('close')
    closefeats_h2=get_feats(closedata[:,4:]) . append('close')
    
    print('************* \n Neutral Hand!')
    time.sleep(2)
    neutraldata=record_some_data
    print('data collected, rest your hand for a moment!')
    neutralfeats_h1=get_feats(neutraldata[:,:4]) . append('neu')
    neutralfeats_h2=get_feats(neutraldata[:,4:]) . append('neu')
    
    print('************* \n Open Hand!')
    time.sleep(2)
    opendata=record_some_data
    print('data collected, rest your hand for a moment!')
    openfeats_h1=get_feats(opendata[:,:4]) . append ('opne')
    openfeats_h2=get_feats(opendata[:,4:]) . append ('oopen')
    
    h1_traindat=[closefeats_h1,neutralfeats_h1,openfeats_h1]
    model_h1=GaussianNB()
    model_h1,acc_h1=train_model(model_h1,h1_traindat)
    
    h2_traindat=[closefeats_h2,neutralfeats_h2,openfeats_h2]
    model_h2=GaussianNB()
    model_h2,acc_h2=train_model(model_h2,h2_traindat)
    
    m.disconnect()
    pyoc_run_split(model_h1, model_h2)