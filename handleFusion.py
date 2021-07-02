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
import scipy as sp
import matplotlib.pyplot as plt
import csv

def normalise_weights(w1,w2):
    wtot = w1+w2
    w1 = w1 / wtot
    w2 = w2 / wtot
    return w1,w2

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
    return np.asarray(fused)

def fuse_mean(mode1,mode2):
    mean=fuse_linweight(mode1,mode2,50,50)
    return mean


def fuse_js_offline(mode1,mode2):
    wjs1,wjs2=get_initial_js(mode1,mode2)
    fused=[]
    prior=fuse_linweight(mode1,mode2,wjs1,wjs2)
    fused.append(prior)
    for instance in range(len(mode1)):
        fusedinstance,wjs1,wjs2=fuse_js_single(mode1,mode2,wjs1,wjs2,prior)
        fused.append(fusedinstance)
    return fused
    
def incorporate_prior(fused,prior):
    read_the_matlab="please"
    return fused

def update_weights(w1,w2,fused):
    return w1,w2
    
def fuse_js_single(mode1,mode2,w1,w2,prior,loud=0):
    fused=prior*((mode1*w1) + (mode2*w2))
    #following line is uncommented for non-temporal
    #fusion(i,:) =  ( testmod1(i,:).*js_w1 ) + ( testmod2(i,:).*js_w2 );
    fused = fused/fused.sum()
    if loud:
        print('m1 dist: ',np.around(mode1,3))
        print('m2 dist: ',np.around(mode2,3))
        print('prior dist: ',np.around(prior,3))
        print('fus dist: ',np.around(fused,3))
    #temporal: it's needed to absorb transitions
    ind = np.argmax(fused);
    if (fused[0,ind] > 0.999):
        _,c = np.shape(mode1);
        fused = [0.001/(c-1)]*c;
        fused[ind] = 0.999;
    width=prior.shape[1]
    return np.reshape(np.asarray(fused),(1,width))

def fuse_js_loop(mode1,mode2,w1,w2):
    fusion = [];
    r,c = np.shape(mode1);
    prior=[1/3]*c;
    i=1
    while i<=r:
        prior = fuse_js_single(mode1[i,:],mode2[i,:],w1,w2,prior)
        fusion.append(prior)
        i+=1
    return fusion

def calc_entropy(probs):
    probs=np.asarray(probs)
    probsq=np.square(probs[probs!=0])
    ents=probsq*np.log(probsq)
    entrop=-sum(ents)
    return entrop
#signal entropy as per the matlab shannon wavelet entropy implementation:
#https://uk.mathworks.com/help/wavelet/ref/wentropy.html#mw_de3a1d48-6b30-49dd-95d1-1e9a01414d75   
#see also:
#https://raphaelvallat.com/entropy/build/html/_modules/entropy/entropy.html

def get_initial_js(mode1,mode2,loud=0):
    ind_train1 = np.argmax(mode1.sum(axis=0))
    ind_train2 = np.argmax(mode2.sum(axis=0))
    # Learning the uncertainty: compute the entropy given the training
    #hmode1_old = sp.stats.entropy(mode1[:,ind_train1])
    #hmode2_old = sp.stats.entropy(mode2[:,ind_train2])
    hmode1 = calc_entropy(mode1[:,ind_train1])
    hmode2 = calc_entropy(mode2[:,ind_train2])
    #print('scipy: ',hmode1_old,'\nmatlab: ',hmode1)
    #print('scipy: ',hmode2_old,'\nmatlab: ',hmode2)
    htot = hmode1+hmode2;
    # weights: normalization/distribution of entropy values
    w1 = 1 - (hmode1 / htot);
    w2 = 1 - (hmode2 / htot);
    w1,w2=normalise_weights(w1,w2)
    #KL divergence P(y|w) and P(w|y)
    # modality1
    KLpyw_m1 = sum(mode1[:,ind_train1]*(mode1[:,ind_train1]/w1));
    KLpwy_m1 = sum(w1*(w1/mode1[:,ind_train1])); 
    # modality2
    KLpyw_m2 = sum(mode2[:,ind_train2]*(mode2[:,ind_train2]/w2));
    KLpwy_m2 = sum(w2*(w2/mode2[:,ind_train2]));
    #JS divergence
    JS_mod1 = (KLpyw_m1 + KLpwy_m1) * 0.5;
    JS_mod2 = (KLpyw_m2 + KLpwy_m2) * 0.5; 
    if np.isinf(JS_mod1):
        JS_mod1=1
    if np.isinf(JS_mod2):
        JS_mod2=1
    # JS weights: normalization
    tot = (JS_mod1 + JS_mod2);
    wjs1 = JS_mod1 / tot;
    wjs2 = JS_mod2 / tot;
    if loud:
        print('w1: ',w1)
        print('w2: ',w2)
        print('JS_mod1: ',JS_mod1)
        print('JS_mod2: ',JS_mod2)
    print('wjs1: ',wjs1)
    print('wjs2: ',wjs2)
    if wjs1<0.00001:
        wjs1=0
    if wjs2<0.00001:
        wjs2=0
    if np.isnan(wjs1):
        nan_weight_report(1,w1,w2,JS_mod1,JS_mod2)
    if np.isnan(wjs2):
        nan_weight_report(2,w1,w2,JS_mod1,JS_mod2)
    return wjs1,wjs2

def nan_weight_report(mode,w1,w2,JS_mod1,JS_mod2):
    print('w',mode,' is nan')
    print('w1: ',w1)
    print('w2: ',w2)
    print('JS_mod1: ',JS_mod1)
    print('JS_mod2: ',JS_mod2)
    raise KeyboardInterrupt

def js_div_w(trainmod1, trainmod2, testmod1, testmod2):

    #preparing the training set to learn the entropy

    # get the winner class (higher prob along frames)
    #vmax_train1 = max(trainmod1.sum())
    ind_train1 = np.argmax(trainmod1.sum())
    #vmax_train2 = max(trainmod2.sum())
    ind_train2 = np.argmax(trainmod2.sum())
    # Learning the uncertainty: compute the entropy given the training
    htrainmod1 = sp.entropy(trainmod1[:,ind_train1])
    htrainmod2 = sp.entropy(trainmod2[:,ind_train2])
    htot = htrainmod1+htrainmod2;
    
    # weights: normalization/distribution of entropy values
    w1 = 1 - (htrainmod1 / htot);
    w2 = 1 - (htrainmod2 / htot);
    w1,w2=normalise_weights(w1,w2)
    
    #[r,c] = size(testmod1);
    #KL divergence P(y|w) and P(w|y)
    # modality1
    KLpyw_m1 = sum(trainmod1[:,ind_train1]*(trainmod1[:,ind_train1]/w1));
    KLpwy_m1 = sum(w1*(w1/trainmod1[:,ind_train1]));
       
    # modality2
    KLpyw_m2 = sum(trainmod2[:,ind_train2]*(trainmod2[:,ind_train2]/w2));
    KLpwy_m2 = sum(w2*(w2/trainmod2[:,ind_train2]));

    #JS divergence
    JS_mod1 = (KLpyw_m1 + KLpwy_m1) * 0.5;
    JS_mod2 = (KLpyw_m2 + KLpwy_m2) * 0.5;
       
    # JS weights: normalization
    tot = (JS_mod1 + JS_mod2);
    js_w1 = JS_mod1 / tot;
    js_w2 = JS_mod2 / tot;
       
    # Fusion
    #get dimension
    fusion = [];
    r,c = np.shape(testmod1);
    prior=[1/3]*c;
    i=1
    while i<=r:
        #temporal
        fusion[i,:] =  prior*((testmod1[i,:]*js_w1) + (testmod2[i,:]*js_w2));
        #following line is uncommented for non-temporal
        #fusion(i,:) =  ( testmod1(i,:).*js_w1 ) + ( testmod2(i,:).*js_w2 );
        fusion[i,:] = fusion[i,:]/sum(fusion[i,:]);
        
        #temporal: it's needed to absorb transitions
        [val, ind] = max(fusion[i,:]);
        if (fusion[i, ind] > 0.99):
            fusion[i,:] = 0.01/(c-1);
            fusion[i, ind] = 0.99;
          
        prior = fusion[i,:];
        i+=1
    return fusion,js_w1,js_w2
#def adaptive_fusion:
 #   try_some
  #  pick_best_strat
    #report strat to top level and then call this again after a while??