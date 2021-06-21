#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 23:28:23 2021

@author: michael

methods for managing Tkinter prompts in EEG/EMG experiments
"""

from tkinter import *
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory
from PIL import ImageTk,Image
import time

class Gesture:
    def __init__(self,label,img):
        self.label=label
        self.rep=0
        self.img=img
    
    def __repr__(self):
        return(f'{self.__class__.__name__}('
               f'{self.label!r}, {self.img!r})')

def new_gest():
    label=str(input("gesture name: "))
    Tk().withdraw()
    img=askopenfilename(initialdir=startpath,title='Select Gesture Diagram')
    gesture=Gesture(label,img)
    return gesture

def setup_default_gests():
    Close = Gesture("close","/home/michael/Documents/Aston/MultimodalFW/prompts/close.jpg")
    Open = Gesture("open","/home/michael/Documents/Aston/MultimodalFW/prompts/open.jpg")
    Neutral = Gesture("neutral","/home/michael/Documents/Aston/MultimodalFW/prompts/space.jpg")
    Grasp = Gesture("grasp","/home/michael/Documents/Aston/MultimodalFW/prompts/grasp.png")
    Lateral = Gesture("lateral","/home/michael/Documents/Aston/MultimodalFW/prompts/lateral.png")
    Tripod = Gesture("tripod","/home/michael/Documents/Aston/MultimodalFW/prompts/tripod.png")
    return [Close,Open,Neutral,Grasp,Lateral,Tripod]
    
def display_prompt(figwin,gesture,gestlist,count):
    print(gesture.label)
    figwin.title('Gesture #'+str(count)+' of '+str(len(gestlist)))
    file=gesture.img
    img=ImageTk.PhotoImage(Image.open(file),master=figwin.canvas)
    figwin.canvas.itemconfig(figwin.img, image=img)
    figwin.update_idletasks()
    figwin.update()

def display_setup(gestlist):
    figwin=Tk()
    figwin.title('Gesture #'+str(0)+' of '+str(len(gestlist)))
    figwin.canvas=Canvas(figwin,width=225,height=175)
    figwin.canvas.pack()
    startfile = "/home/michael/Documents/Aston/MultimodalFW/prompts/space.jpg"
    startimg=ImageTk.PhotoImage(Image.open(startfile),master=figwin.canvas)
    figwin.img=figwin.canvas.create_image(20,20,anchor=NW,image=startimg)
    figwin.update_idletasks()
    figwin.update()
    time.sleep(1)
    return figwin