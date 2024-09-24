# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:12:40 2021

@author: pritcham

module for OS-dependent parameters
"""

from sys import platform
import os
#import glob    #unused approach that allows us to be extension-agnostic if we have mixed jpgs and pngs

gestures_to_idx = {'close':1.,'open':2.,'grasp':3.,'lateral':4.,'tripod':5.,'neutral':0.,'cylindrical':6.,'spherical':7.,'lumbrical':8.,'rest':9.}
idx_to_gestures = {1.:'close',2.:'open',3.:'grasp',4.:'lateral',5.:'tripod',0.:'neutral',6.:'cylindrical',7.:'spherical',8.:'lumbrical',9.:'rest'}

gestures_to_idx_binary = {'neutral':0.,'cylindrical':1.,'spherical':1.,'lumbrical':1.,'rest':0.}
idx_to_gestures_binary = {1.:'grasp',0.:'rest'}

gestures_to_idx_deeplearn = {'cylindrical':0.,'lumbrical':1.,'rest':2.,'spherical':3.}
idx_to_gestures_deeplearn = {0.:'cylindrical',1.:'lumbrical',2.:'rest',3.:'spherical'}

runletter_to_num = {'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'j':10,'k':11,'l':12,'m':13,'n':14,'o':15,'p':16}
runnum_to_letter = {1:'a',2:'b',3:'c',4:'d',5:'e',6:'f',7:'g',8:'h',9:'i',10:'j',11:'k',12:'l',13:'m',14:'n',15:'o',16:'p'}

currentpath=os.path.dirname(__file__)

if platform =='win32':
    #windows
    path=r'C:\Users\pritcham\Documents\python\data-dump'
    prompts_dir="C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts"
    #prompt_neut=glob.glob("C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/neutral.*")[0]
    #stackoverflow.com/questions/19824598/open-existing-file-of-unknown-extension/19824624
    prompt_neut="C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/neutral.jpg"
    prompt_close="C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/close.jpg"
    prompt_open="C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/open.jpg"
    prompt_grasp="C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/grasp.jpg"
    prompt_lateral="C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/lateral.jpg"
    prompt_tripod="C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/tripod.jpg"
    #gen_trainmat_spec_SpellLoc="C:/Users/pritcham/Documents/python/labelstoClass/labelstoClassEMGSpell.py"
    gen_trainmat_spec_SpellLoc=os.path.join(currentpath,'labelstoClassEMGSpell.py')

    
    emg_set_path_for_system_tests='C:/Users/pritcham/Documents/mm-framework/working_dataset/devset_EMG/featsEMG.csv'
    eeg_set_path_for_system_tests='C:/Users/pritcham/Documents/mm-framework/working_dataset/devset_EEG/featsEEG.csv'
    
    emg_waygal='C:/Users/pritcham/Documents/mm-framework/WAYGAL/WaygalAllEMGFeats.csv'
    eeg_waygal='C:/Users/pritcham/Documents/mm-framework/WAYGAL/WayGalAllEEGFeats.csv'
    eeg_32_waygal='C:/Users/pritcham/Documents/mm-framework/multimodal-framework/lit_data_expts/waygal/datasets/FullChEEGFeats.csv'
    
    waygal_results_dir='lit_data_expts/waygal/results/'
    
    jeong_EEGdir='H:/Jeong11tasks_data/RawCSVs/'
    eeg_jeong_feats='H:/Jeong11tasks_data/jeongEEGfeats.csv'
    
    jeongCSP_EEGdir='H:/Jeong11tasks_data/CSP_CSVs/'
    eeg_jeongCSP_feats='H:/Jeong11tasks_data/jeongCSPEEGfeats.csv'
    
    jeongSyncCSP_EEGdir='H:/Jeong11tasks_data/Synced_CSP_CSVs/'
    eeg_jeongSyncCSP_feats='H:/Jeong11tasks_data/jeongSyncCSPEEGfeats.csv'
    
    jeongSyncRawEEGdir='H:/Jeong11tasks_data/RawEEGnoCSP_synced/'
    eeg_jeongSyncRaw_feats='H:/Jeong11tasks_data/jeongSyncRawEEGfeats.csv'
    
    jeong_EMGdir='H:/Jeong11tasks_data/EMG/Raw_EMGCSVs/'
    jeong_EMGfeats='H:/Jeong11tasks_data/jeong_EMGfeats.csv'
    
    jeong_noCSP_WidebandDir='H:/Jeong11tasks_data/EEGnoCSP_wideband/'
    jeong_noCSP_WidebandFeats='H:/Jeong11tasks_data/EEGnoCSP_WidebandFeats.csv'
    
    jeong_eeg_noholdout=r"H:\Jeong11tasks_data\final_dataset\jeongEEG_noholdout.csv"
    jeong_emg_noholdout=r"H:\Jeong11tasks_data\final_dataset\jeongEMG_noholdout.csv"

    jeong_EMGfeats_server=os.path.join(currentpath,'lit_data_expts/jeong/datasets/jeong_EMGfeats.csv')

    jeong_EEGfeats_server=os.path.join(currentpath,'lit_data_expts/jeong/datasets/jeong_EEGfeatsCSPSync.csv')
    
    jeong_RawEEGfeats_server=os.path.join(currentpath,'lit_data_expts/jeong/datasets/jeong_RawEEGfeatsSync.csv')
    
    jeong_noCSP_WidebandFeats_server=os.path.join(currentpath,'lit_data_expts/jeong/datasets/jeongEEGnoCSP_WBFeats.csv')
    
    jeong_EMGnoHO_server=os.path.join(currentpath,'lit_data_expts/jeong/datasets/jeongEMG_noholdout.csv')
    jeong_EEGnoHO_server=os.path.join(currentpath,'lit_data_expts/jeong/datasets/jeongEEG_noholdout.csv')
    
    jeong_results_dir='lit_data_expts/jeong/results/'
    
    emgLOOfeatpath=os.path.join(currentpath, 'lit_data_expts/jeong/datasets/emg_LOO_feats_15pct.csv')
    eegLOOfeatpath=os.path.join(currentpath, 'lit_data_expts/jeong/datasets/eeg_LOO_feats_L1_88.csv')
    jointemgeegLOOfeatpath=os.path.join(currentpath, 'lit_data_expts/jeong/datasets/joint_LOO_feats_L1_176.csv')
    
    
    jeong_eeg_filtfilt_dir = 'H:/Jeong11tasks_data/FiltFilt_EEG/'
    jeong_eeg_filtfilt_path = 'H:/Jeong11tasks_data/jeong_FiltFilt_EEG_feats.csv'
    
elif platform == 'darwin':
    #mac
    path='macpath'
elif platform == 'linux' or 'linux32':
    #linux
    path="/home/michael/Documents/Aston/MultimodalFW/pyoc_data_dump/"
    prompts_dir="/home/michael/Documents/Aston/MultimodalFW/prompts/"
    prompt_neut="/home/michael/Documents/Aston/MultimodalFW/prompts/space.jpg"
    prompt_close="/home/michael/Documents/Aston/MultimodalFW/prompts/close.jpg"
    prompt_open="/home/michael/Documents/Aston/MultimodalFW/prompts/open.jpg"
    prompt_grasp="/home/michael/Documents/Aston/MultimodalFW/prompts/grasp.png"
    prompt_lateral="/home/michael/Documents/Aston/MultimodalFW/prompts/lateral.png"
    prompt_tripod="/home/michael/Documents/Aston/MultimodalFW/prompts/tripod.png"
    #gen_trainmat_spec_SpellLoc="/home/michael/github/labelstoClass/labelstoClassEMGSpell.py"
    gen_trainmat_spec_SpellLoc=os.path.join(currentpath,'labelstoClassEMGSpell.py')
    
    emg_set_path_for_system_tests='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EMG/featsEMG.csv'
    #eeg_set_path_for_system_tests='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EEG/featsEEG.csv'
    '''new EEG featset based on new sig proc pipeline'''
    #eeg_set_path_for_system_tests='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EEG/featsEEGNewDec.csv'
    eeg_set_path_for_system_tests='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EEG/featsEEGNewDecImpulseKill.csv'
    
    emg_path_waygal_4812='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/P4_8_12/P4812_emg_Feats.csv'
    eeg_path_waygal_4812='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/P4_8_12/P4812_eeg_Feats.csv'
    
    #eeg_waygal='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/All_CSVs/WayGalAllEEGFeats.csv'
    #emg_waygal='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/All_CSVs/WaygalAllEMGFeats.csv'
    eeg_waygal=os.path.join(currentpath,'lit_data_expts/waygal/datasets/waygalAllEEGFeats.csv')
    emg_waygal=os.path.join(currentpath,'lit_data_expts/waygal/datasets/waygalAllEMGFeats.csv')
    eeg_32_waygal=os.path.join(currentpath,'lit_data_expts/waygal/datasets/FullChEEGFeats.csv')
    
    waygal_results_dir='lit_data_expts/waygal/results/'
    
    all_channel_waygal_EEG=os.path.normpath(os.path.join(currentpath,'../waygal-raw-eeg/Full_Channel_EEG'))
    
    jeong_EMGfeats='H:/Jeong11tasks_data/jeong_EMGfeats.csv'
    eeg_jeongSyncCSP_feats='H:/Jeong11tasks_data/jeongSyncCSPEEGfeats.csv'
    eeg_jeongSyncRaw_feats='H:/Jeong11tasks_data/jeongSyncRawEEGfeats.csv'
    jeong_noCSP_WidebandDir='H:/Jeong11tasks_data/EEGnoCSP_wideband/'
    jeong_noCSP_WidebandFeats='H:/Jeong11tasks_data/EEGnoCSP_WidebandFeats.csv'
    
    jeong_eeg_noholdout=r"H:\Jeong11tasks_data\final_dataset\jeongEEG_noholdout.csv"
    jeong_emg_noholdout=r"H:\Jeong11tasks_data\final_dataset\jeongEMG_noholdout.csv"
    
    jeong_EMGfeats_server=os.path.join(currentpath,'lit_data_expts/jeong/datasets/jeong_EMGfeats.csv')

    jeong_EEGfeats_server=os.path.join(currentpath,'lit_data_expts/jeong/datasets/jeong_EEGfeatsCSPSync.csv')
    
    jeong_RawEEGfeats_server=os.path.join(currentpath,'lit_data_expts/jeong/datasets/jeong_RawEEGfeatsSync.csv')
    
    jeong_noCSP_WidebandFeats_server=os.path.join(currentpath,'lit_data_expts/jeong/datasets/jeongEEGnoCSP_WBFeats.csv')
    
    jeong_EMGnoHO_server=os.path.join(currentpath,'lit_data_expts/jeong/datasets/jeongEMG_noholdout.csv')
    jeong_EEGnoHO_server=os.path.join(currentpath,'lit_data_expts/jeong/datasets/jeongEEG_noholdout.csv')
    
    jeong_results_dir='lit_data_expts/jeong/results/'
    
    emgLOOfeatpath=os.path.join(currentpath, 'lit_data_expts/jeong/datasets/emg_LOO_feats_15pct.csv')
    eegLOOfeatpath=os.path.join(currentpath, 'lit_data_expts/jeong/datasets/eeg_LOO_feats_L1_88.csv')
    jointemgeegLOOfeatpath=os.path.join(currentpath, 'lit_data_expts/jeong/datasets/joint_LOO_feats_L1_176.csv')
    
    
    jeong_eeg_filtfilt_dir = 'H:/Jeong11tasks_data/FiltFilt_EEG/'
    jeong_eeg_filtfilt_path = 'H:/Jeong11tasks_data/jeong_FiltFilt_EEG_feats.csv'
