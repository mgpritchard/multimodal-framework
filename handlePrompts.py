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
    Close = Gesture("close","C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/close.jpg")
    Open = Gesture("open","C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/open.jpg")
    Neutral = Gesture("neutral","C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/space.jpg")
    Grasp = Gesture("grasp","C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/grasp.png")
    Lateral = Gesture("lateral","C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/lateral.png")
    Tripod = Gesture("tripod","C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/tripod.png")
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
    startfile = "C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/space.jpg"
    startimg=ImageTk.PhotoImage(Image.open(startfile),master=figwin.canvas)
    figwin.img=figwin.canvas.create_image(20,20,anchor=NW,image=startimg)
    figwin.update_idletasks()
    figwin.update()
    time.sleep(1)
    return figwin

def display_predictions_setup():
    predwin=Tk()
    predwin.title('Predicted gesture')
    width=225*2
    height=175*2
    predwin.canvas=Canvas(predwin,width=width,height=height)
    predwin.canvas.pack()
    startfile = "C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts/space.jpg"
    startimg=ImageTk.PhotoImage(Image.open(startfile),master=predwin.canvas)
    predwin.img1=predwin.canvas.create_image(20,20,anchor=NW,image=startimg)
    predwin.w1=predwin.canvas.create_text(20,int(height/2),anchor=NW,text="No weight yet")
    predwin.img2=predwin.canvas.create_image(width-20,20,anchor=NE,image=startimg)
    predwin.w2=predwin.canvas.create_text(width-20,int(height/2),anchor=NE,text="No weight yet")
    predwin.imgfusion=predwin.canvas.create_image(int(width/2),height-20,anchor=S,image=startimg)
    predwin.update_idletasks()
    predwin.update()
    time.sleep(1)
    return predwin

def fetch_img(gesture):
    imgsource="C:/Users/pritcham/Documents/python/mm-prompts/multimodal-prompts"
    if gesture=='neutral':
        gesture='space'
    if gesture.lower() in ['open','close','space']:
        ext='.jpg'
    elif gesture.lower() in ['grasp','lateral','tripod']:
        ext='.png'
    else:
        gesture='space';ext='.jpg'
    file=imgsource+gesture.lower()+ext
    return file

def display_predictions(predwin,gesture1,gesture2,gesturef):
    predwin.title('Predicted gesture:'+gesturef)
    file1=fetch_img(gesture1)
    file2=fetch_img(gesture2)
    filef=fetch_img(gesturef)
    img1=ImageTk.PhotoImage(Image.open(file1),master=predwin.canvas)
    img2=ImageTk.PhotoImage(Image.open(file2),master=predwin.canvas)
    imgf=ImageTk.PhotoImage(Image.open(filef),master=predwin.canvas)
    predwin.canvas.itemconfig(predwin.img1, image=img1)
    predwin.canvas.itemconfig(predwin.img2, image=img2)
    predwin.canvas.itemconfig(predwin.imgfusion, image=imgf)
    predwin.update_idletasks()
    predwin.update()

def display_preds_and_weights(predwin,gesture1,gesture2,gesturef,w1,w2):
    predwin.title('Predicted gesture:'+gesturef)
    file1=fetch_img(gesture1)
    file2=fetch_img(gesture2)
    filef=fetch_img(gesturef)
    img1=ImageTk.PhotoImage(Image.open(file1),master=predwin.canvas)
    img2=ImageTk.PhotoImage(Image.open(file2),master=predwin.canvas)
    imgf=ImageTk.PhotoImage(Image.open(filef),master=predwin.canvas)
    predwin.canvas.itemconfig(predwin.img1, image=img1)
    predwin.canvas.itemconfig(predwin.img2, image=img2)
    predwin.canvas.itemconfig(predwin.imgfusion, image=imgf)
    label1="Weight for mode 1: "+format(w1,'.3f')
    label2="Weight for mode 2: "+format(w2,'.3f')
    predwin.canvas.itemconfig(predwin.w1,text=label1)
    predwin.canvas.itemconfig(predwin.w2,text=label2)
    predwin.update_idletasks()
    predwin.update()
    
def display_weights(predwin,w1,w2):
    label1="Weight for mode 1: "+format(w1,'.3f')
    label2="Weight for mode 2: "+format(w2,'.3f')
    predwin.canvas.itemconfig(predwin.w1,text=label1)
    predwin.canvas.itemconfig(predwin.w2,text=label2)
    img1=predwin.canvas.itemcget(predwin.img1,"image")
    predwin.canvas.itemconfig(predwin.img1,image=img1.filename)
    predwin.canvas.itemconfig(predwin.img2)
    predwin.canvas.itemconfig(predwin.imgfusion)
    predwin.update_idletasks()
    predwin.update()