# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:12:40 2021

@author: pritcham

module for OS-dependent parameters
"""

from sys import platform

if platform =='win32':
    #windows
    path=r'C:\Users\pritcham\Documents\python\data-dump'
    prompts_dir="C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts"
    prompt_neut="C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/space.jpg"
    prompt_close="C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/close.jpg"
    prompt_open="C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/open.jpg"
    prompt_grasp="C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/grasp.png"
    prompt_lateral="C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/lateral.png"
    prompt_tripod="C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/tripod.png"
    gen_trainmat_spec_SpellLoc="C:/Users/pritcham/Documents/python/labelstoClass/labelstoClassEMGSpell.py"
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