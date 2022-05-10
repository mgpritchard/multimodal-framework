#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 22:08:53 2021

@author: pritcham

script for recording gestures as part of multimodal experiment
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from tkinter import *
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory
from PIL import ImageTk,Image
from queue import Queue
import logging
from threading import Thread
import time
from handleEMG import *
from handleEEG import *
from handleML import *
from handleOffline import *
from handleFusion import *
from handlePrompts import *
import pickle
import random
import brainflow
import traceback
import params
import matplotlib.pyplot as plt
from serial import SerialException

def record_gesture(path,gesture,duration,pptid,rep):
    pyoc_record_fixed_time(path, duration)
    rename_latest_emg(path, pptid, gesture.label, gesture.rep)

def show_and_record(gesture,pptid,path,duration,figwin,gestlist,count,boardEEG,m,rest):
    boardEEG.start_stream()
    time.sleep(1) #ensure eeg is recording before prompt is shown
    display_prompt(figwin,gesture,gestlist,count)#
    #plot_prompt(gesture)
    '''plot_update(figwin[0],figwin[1],gesture)'''
    boardEEG.insert_marker(1) #marker in eegstream to denote prompt being shown
    #pyoc_record_fixed_time(path, duration) #emg recording has to start after
            #prompt as it is a blocking method; thread not released until end
    try:
        pyoc_record_no_init(path, duration, m)
    except SerialException as e:
        print(e)
        print(traceback.format_exc())
        m.disconnect()
        delete_latest_emg(path)
        print("\n\n*****Retrying...*****\n\n") 
        #garbage=boardEEG.get_board_data()
        #boardEEG.stop_stream()
        time.sleep(1)
        #boardEEG.start_stream()
        boardEEG.insert_marker(1)
        pyoc_record_no_init(path,duration,m)
    except PermissionError as e:
        print(e)
        print(traceback.format_exc())
        m.disconnect()
        delete_latest_emg(path)
        print("\n\n*****Retrying...*****\n\n") 
        #garbage=boardEEG.get_board_data()
        #boardEEG.stop_stream()
        time.sleep(1)
        #boardEEG.start_stream()
        boardEEG.insert_marker(1)
        pyoc_record_no_init(path,duration,m)
    rename_latest_emg(path,pptid,gesture.label,gesture.rep)
    dataEEG = boardEEG.get_board_data()  # get all data and remove it from internal buffer
    boardEEG.stop_stream()
    save_EEG(dataEEG,path,gesture,pptid)
    gesture.rep+=1
    '''plot_rest(figwin[0],figwin[1])'''
    #plt.close()
    
    #would be good to rewrite such that recording carries on for x seconds
    #following the rest prompt, with another marker... but again emg recording
    #is a blocking method. only option may be to spin up another thread?

def record_continuous(pptid,path,numreps):
    try:
        boardEEG=setup_bf("unicorn") #"unicorn"
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        kill_bf("unicorn")
        boardEEG=setup_bf()  
    
    m = pyoc_init()

    filename_EMG=path+'/'+pptid+'-continuous'+'.csv'
    #destfile=(path+"/"+filename_EMG)
    with open(filename_EMG,'w',newline='') as csvfile:
        emgwriter=csv.writer(csvfile, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)

        def proc_emg(emg, moving, times=[]):
            ## print framerate of received data
            times.append(time.time())
            if len(times) > 20:
                #print((len(times) - 1) / (times[-1] - times[0]))
                times.pop(0)
        
        gestlist=[]
        gests=setup_default_gests()
        rest = Gesture("rest",params.prompt_neut)
        for gest in gests:
            gestlist.extend(gest for i in range(numreps))
        random.shuffle(gestlist)
        figwin=display_setup(gestlist)#
        count=0
        
        duration=random.randint(400,500)/100
        resttime=random.randint(1000,1200)/100
        resting=1
            
        boardEEG.start_stream()
        time.sleep(1) #ensure eeg is recording before prompt is shown
        
        m.add_emg_handler(proc_emg)
        m.connect()

        try:
            tstart=time.time()
            while count < len(gestlist):
                currentemg=m.run_report(1)
                print(currentemg)
                if currentemg is not None:
                    #print(currentemg)
                    emgwrite=list(currentemg)
                    emgwrite.insert(0,(int(round(time.time() * 1000))))
                    emgwrite=tuple(emgwrite)
                    print(emgwrite)                #maybe use to debug?
                    emgwriter.writerow(emgwrite)    #currently writing as a string of '[20 20 20 20 20 20 20 20]'
                #else:
                 #   continue

                if resting==1:
                    if time.time() >= tstart+resttime:
                        #print('start: ',tstart,'\nend: ',time.time(),'\nperiod: ',period)                                                         
                        resting=0
                        gesture=gestlist[count]
                        display_prompt(figwin,gesture,gestlist,count)#
                        boardEEG.insert_marker(1) 
                        count+=1
                        gesture.rep+=1
                        duration=random.randint(400,500)/100                    
                        tstart=time.time()
                        
                elif resting==0:
                    if time.time() >= tstart+duration:
                        resting=1
                        display_prompt(figwin,rest,gestlist,count)#
                        boardEEG.insert_marker(1)
                        resttime=random.randint(1000,1200)/100
                        tstart=time.time()
                    
                    
            dataEEG = boardEEG.get_board_data()  # get all data and remove it from internal buffer
            boardEEG.stop_stream()
            filename_EEG=path+'/'+pptid+'-continuous'+'_EEG.csv'
            DataFilter.write_file(dataEEG, filename_EEG, 'w')
          
            m.disconnect()
            boardEEG.release_session()  #NEEDS TO REACH TO AVOID KERNEL RESTART!
            print('All done! Thanks for your contribution')
            figwin.destroy()
             
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            m.disconnect()
            figwin().destroy()
            
            dataEEG = boardEEG.get_board_data()  # get all data and remove it from internal buffer
            boardEEG.stop_stream()
            filename_EEG=path+'/'+pptid+'-continuous'+'_EEG.csv'
            DataFilter.write_file(dataEEG, filename_EEG, 'w')
            boardEEG.release_session()
        finally:
            print()
 

def record_exptl(pptid,path,duration,numreps,resttime):
    try:
        boardEEG=setup_bf("unicorn") #"unicorn"
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        kill_bf("unicorn")
        boardEEG=setup_bf()
        
    try:
        delete_latest_emg(path)
        gestlist=[]
        gests=setup_default_gests()
        rest = Gesture("rest",params.prompt_neut)
        for gest in gests:
            gestlist.extend(gest for i in range(numreps))
        random.shuffle(gestlist)
        figwin=display_setup(gestlist)#
        '''[plt,fig,ax]=plot_init(rest)
        figwin=[plt,fig,ax]'''
        count=0
        saveloc="temp_gestlist"
        with open(saveloc,"wb") as savelist:
            pickle.dump(gestlist,savelist)
        m=pyoc_init()
        for gesture in gestlist:
            count+=1
            duration=random.randint(400,500)/100
            resttime=random.randint(1000,1200)/100
            print('dur: ',duration)
            show_and_record(gesture,pptid,path,duration,figwin,gestlist,count,boardEEG,m,rest)
            '''tstart=time.time()      #swap out with above if testing timing
            tend=tstart+duration
            print('start: ',tstart,'\n end: ',tend)
            time.sleep(duration)'''
            display_prompt(figwin,rest,gestlist,count)#
            #plot_update(figwin[0],figwin[1],rest)
            print('rest now',' #',count)
            '''trest=resttime/2
            while trest>0:
                #print('+',end="")
                time.sleep(1)
                trest=trest-1'''
            time.sleep(resttime)
            print('rested')
            #plt.close()
            #if count>=3:#
            #    raise ValueError#
    except Exception as e:
        boardEEG.release_session()
        figwin.destroy()
        print(e)
        print(traceback.format_exc())
        print('\nfound an error, lets retry\n')#
        m.bt.ser.close() #
        del m#
        raise e
    else:
        boardEEG.release_session()  #NEEDS TO REACH TO AVOID KERNEL RESTART!
        print('All done! Thanks for your contribution')
        figwin.destroy()
  


  
def resume_exptl(pptid,path,duration,numreps,resttime,idx,m=0):
    try:
        boardEEG=setup_bf("unicorn") #"unicorn"
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        kill_bf("unicorn")
        boardEEG=setup_bf()
        
    try:
        delete_latest_emg(path)
        rest = Gesture("rest",params.prompt_neut)
        count=idx
        saveloc="temp_gestlist"
        with open(saveloc,"rb") as savelist:            #This version picks up where you left off!
            gestlist=pickle.load(savelist)
        figwin=display_setup(gestlist)#
        
        if m is 0:
            m=pyoc_init()
        while count < len(gestlist):
            gesture = gestlist[count]
            count+=1
            duration=random.randint(400,500)/100
            resttime=random.randint(1000,1200)/100
            print('dur: ',duration)
            show_and_record(gesture,pptid,path,duration,figwin,gestlist,count,boardEEG,m,rest)
            
            display_prompt(figwin,rest,gestlist,count)#
            #plot_update(figwin[0],figwin[1],rest)
            print('rest now',' #',count)
            
            time.sleep(resttime)
            print('rested')
            #plt.close()
            #if count==6:#
            #    raise ValueError#
    except Exception as e:
        boardEEG.release_session()
        figwin.destroy()
        print(e)
        print(traceback.format_exc())
        m.bt.ser.close() #
        del m#
        raise e
    else:
        boardEEG.release_session()  #NEEDS TO REACH TO AVOID KERNEL RESTART!
        print('All done! Thanks for your contribution')
        '''something_to_delete_savedlist'''
        figwin.destroy()
    

def quickplot(data,label):
    fig,ax=plt.subplots()
    ax.plot(data[:,0],data[:,1])
    ax.set(title=label)
    plt.show()

if __name__ == '__main__':
    resuming=input("resuming after a crash? [y/n] ")
    pptid=input('particpant id: ')
    startpath=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    #myo_connect_setup()
    #record_exptl(pptid)
    '''
    Tk().withdraw()
    path=askdirectory(initialdir=startpath,title='where to dump EMG?')
    '''
    #path='C:\AstonApps\Solidworks2019SP3'
    path=params.path
    duration=5
    numreps=5
    resttime=1  #randomised 10-12
    gesture='testing'       

    if resuming is not "y" and resuming is not "Y":
        try:
            record_continuous(pptid,path,numreps)
        except Exception as e:
            print(traceback.format_exc())
            #print("\nattempting to auto-resume\n")
            try:
                pptid=input('participant id: ')
                #lastind=int(input("last successful gesture #: "))
                #resume_exptl(pptid,path,duration,numreps,resttime,idx=lastind+1)#,m=m)
            except Exception as e:
                print("\nattempting to auto-resume again\n")
                #pptid=input('participant id: ')
                #lastind=int(input("last successful gesture #: "))
                #resume_exptl(pptid,path,duration,numreps,resttime,idx=lastind+1)#,m=m)
    elif resuming is "y" or resuming is "Y":
        lastind=int(input("last successful gesture #: "))
        #resume_exptl(pptid,path,duration,numreps,resttime,idx=lastind+1)

    #for i in range(3):
     #   record_gesture(path, gesture, duration, pptid, (i+1))
    #pyoc_record_fixed_time(path, 15)
    #rename_latest_emg(path, pptid, 'testing', 1)
    #pyoc_record_fixed_time(path, 15)
    #rename_latest_emg(path, pptid, 'testing', 2)
