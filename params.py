# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:12:40 2021

@author: pritcham

module for OS-dependent parameters
"""

from sys import platform
#import glob    #unused approach that allows us to be extension-agnostic if we have mixed jpgs and pngs

gestures_to_idx = {'close':1.,'open':2.,'grasp':3.,'lateral':4.,'tripod':5.,'neutral':0.}
idx_to_gestures = {1.:'close',2.:'open',3.:'grasp',4.:'lateral',5.:'tripod',0.:'neutral'}

runletter_to_num = {'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'j':10,'k':11,'l':12,'m':13,'n':14,'o':15,'p':16}
runnum_to_letter = {1:'a',2:'b',3:'c',4:'d',5:'e',6:'f',7:'g',8:'h',9:'i',10:'j',11:'k',12:'l',13:'m',14:'n',15:'o',16:'p'}

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
    gen_trainmat_spec_SpellLoc="C:/Users/pritcham/Documents/python/labelstoClass/labelstoClassEMGSpell.py"
    
    emg_set_path_for_system_tests='C:/Users/pritcham/Documents/mm-framework/working_dataset/devset_EMG/featsEMG.csv'
    eeg_set_path_for_system_tests='C:/Users/pritcham/Documents/mm-framework/working_dataset/devset_EEG/featsEEG.csv'
    
    emg_waygal='C:/Users/pritcham/Documents/mm-framework/WAYGAL/WaygalAllEMGFeats.csv'
    eeg_waygal='C:/Users/pritcham/Documents/mm-framework/WAYGAL/WayGalAllEEGFeats.csv'
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
    gen_trainmat_spec_SpellLoc="/home/michael/github/labelstoClass/labelstoClassEMGSpell.py"
    
    emg_set_path_for_system_tests='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EMG/featsEMG.csv'
    #eeg_set_path_for_system_tests='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EEG/featsEEG.csv'
    '''new EEG featset based on new sig proc pipeline'''
    #eeg_set_path_for_system_tests='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EEG/featsEEGNewDec.csv'
    eeg_set_path_for_system_tests='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EEG/featsEEGNewDecImpulseKill.csv'
    
    emg_path_waygal_4812='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/P4_8_12/P4812_emg_Feats.csv'
    eeg_path_waygal_4812='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/P4_8_12/P4812_eeg_Feats.csv'
    
    eeg_waygal='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/All_CSVs/WayGalAllEEGFeats.csv'
    emg_waygal='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/All_CSVs/WaygalAllEMGFeats.csv'