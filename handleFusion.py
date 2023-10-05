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
import params
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import handleML as ml
import random
#try divergence BETWEEN the two modes?? how to distribute this across them?

def setup_onehot(classlabels):
    labels=[params.idx_to_gestures[label] for label in classlabels]
    options=np.array(labels).reshape(-1,1)
    ohe_dense=OneHotEncoder(sparse=False)
    ohe_dense.fit_transform(options)
    return ohe_dense    
    
def encode_preds_onehot(preds,encoder):
    #preds=[1,2,3,1,0,1,0,1]
    preds_labelled=np.array([params.idx_to_gestures[pred] for pred in preds],dtype=object).reshape(-1,1)
    output=encoder.transform(preds_labelled)
    #cols=np.hsplit(output,output.shape[1])
    return output

def train_catNB_fuser(mode1,mode2,targets):
    '''maybe interesting alternative? https://github.com/phamdinhthang/fusion_naive_bayes'''
    model=CategoricalNB()
    train=np.column_stack([mode1,mode2])
    #train=train_data.values[:,:-1]
    #targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model

def bayesian_fusion(fuser,onehot,predlist_emg,predlist_eeg,classlabels):
    onehot_pred_emg=encode_preds_onehot(predlist_emg,onehot)
    onehot_pred_eeg=encode_preds_onehot(predlist_eeg,onehot)
    fusion_distros=ml.prob_dist(fuser,np.column_stack([onehot_pred_emg,onehot_pred_eeg]))
    fusion_preds=[]
    for distro in fusion_distros:
        pred_fusion=ml.pred_from_distro(classlabels,distro)
       # distrolist_fusion.append(distro_fusion)
        fusion_preds.append(pred_fusion) 
    return fusion_preds
'''
def train_svm_fuser(mode1,mode2,targets,args):
    train=np.column_stack([mode1,mode2])
    kernel=args['kernel']
    C=args['svm_C']
    gamma=args['gamma']
    if kernel=='linear':
        model=ml.SVC(C=C,kernel=kernel,probability=True) #possible need to fix random_state as predict is called multiple times?
    else:
        model=ml.SVC(C=C,kernel=kernel,gamma=gamma,probability=True)
    model.fit(train.astype(np.float64),targets)
    return model
'''
def train_svm_fuser(mode1,mode2,targets,args):
    train=np.column_stack([mode1,mode2])
    C=args['svm_C']
    model=ml.LinearSVC(C=C,dual=False)
    model.fit(train.astype(np.float64),targets)
    return model

def svm_fusion(fuser,onehot,predlist_emg,predlist_eeg,classlabels):
    if onehot is not None:
        predlist_emg=encode_preds_onehot(predlist_emg,onehot)
        predlist_eeg=encode_preds_onehot(predlist_eeg,onehot)
    fusion_preds=fuser.predict(np.column_stack([predlist_emg,predlist_eeg]))
    return fusion_preds

def train_lda_fuser(mode1,mode2,targets,args):
    train=np.column_stack([mode1,mode2])
    solver=args['LDA_solver']
    shrinkage=args['shrinkage']
    if solver == 'svd':
        model=LinearDiscriminantAnalysis(solver=solver)
    else:
        model=LinearDiscriminantAnalysis(solver=solver,shrinkage=shrinkage)
    model.fit(train.astype(np.float64),targets)
    return model

def lda_fusion(fuser,onehot,predlist_emg,predlist_eeg,classlabels):
    if onehot is not None:
        predlist_emg=encode_preds_onehot(predlist_emg,onehot)
        predlist_eeg=encode_preds_onehot(predlist_eeg,onehot)
    fusion_preds=fuser.predict(np.column_stack([predlist_emg,predlist_eeg]))
    return fusion_preds

def train_rf_fuser(mode1,mode2,targets,args):
    train=np.column_stack([mode1,mode2])
    n_trees=args['n_trees']
    max_depth=args['max_depth']
    model=ml.RandomForestClassifier(n_estimators=n_trees,max_depth=max_depth)
    model.fit(train.astype(np.float64),targets)
    return model

def rf_fusion(fuser,onehot,predlist_emg,predlist_eeg,classlabels):
    if onehot is not None:
        predlist_emg=encode_preds_onehot(predlist_emg,onehot)
        predlist_eeg=encode_preds_onehot(predlist_eeg,onehot)
    fusion_preds=fuser.predict(np.column_stack([predlist_emg,predlist_eeg]))
    return fusion_preds

def reward_pattern_match():
    pass
    #assess trace of class probability. reward static high or low or
    #clear binary decisions (i.e. goes high to low and stays there)
    #essentially measuring goodness of fit of pre-prepped 'pattern' curves
    #against the data.
    #could to RMSError or Sum of Sq Errors, all based on residuals
    #http://www.math.utah.edu/~palais/pcr/spike/Evaluating%20the%20Goodness%20of%20Fit%20--%20Fitting%20Data%20(Curve%20Fitting%20Toolbox)%20copy.html
    #wold need both rising and falling edge patters but ALSO the same shifted
    #horizontally as we don't know when in the frame the class has changed
    #i think we assume that in a one-frame space it won't genuinely change
    #class twice!
    #each rising/falling edge is functionally a sigmoid

def normalise_weights(w1,w2):
    wtot = w1+w2
    w1 = w1 / wtot
    w2 = w2 / wtot
    return w1,w2

def fuse_max(mode1,mode2):
    if max(mode1)>=max(mode2):
        fused=mode1
    else:
        fused=mode2
    return fused

def fuse_max_arr(l1, l2):
    return np.array([a if max(a)>max(b) else random.choice([a,b]) if max(a)==max(b) else b for a, b in zip(l1, l2)])

def fuse_conf(mode1,mode2):
    fused=[]
    for instance in range(len(mode1)):
        '''if max(mode1[instance,:])>max(mode2[instance,:]):
            fused.append(mode1)
        else:
            fused.append(mode2)'''
        fused.append(fuse_max(mode1[instance,:],mode2[instance,:]))
    return fused

def fuse_linweight(mode1,mode2,weight1,weight2):
    #total_weight=weight1+weight2
    #weighted_1=(mode1*weight1)/total_weight
    #weighted_2=(mode2*weight2)/total_weight
    #fused=weighted_1+weighted_2
    weight1,weight2=normalise_weights(weight1, weight2)
    fused=(mode1*weight1)+(mode2*weight2)
    return np.asarray(fused)

def fuse_select(emg,eeg,args):
    alg=args['fusion_alg']
    if type(alg) is dict:
        alg=alg['fusion_alg_type']
    if alg=='mean':
        fusion = fuse_mean(emg,eeg)
    elif alg=='3_1_emg':
        fusion = fuse_linweight(emg,eeg,75,25)
    elif alg=='3_1_eeg':
        fusion = fuse_linweight(emg,eeg,25,75)
    elif alg=='opt_weight':
        fusion = fuse_linweight(emg,eeg,100-args['eeg_weight_opt'],args['eeg_weight_opt'])
    elif alg=='highest_conf':
        fusion = fuse_max_arr(emg,eeg)
    elif alg=='bayes':
        '''bayesian fusion is not done here, just keeping system happy'''
        fusion = fuse_mean(emg,eeg)
    elif alg=='featlevel':
        '''feature level fusion is not done here, just keeping system happy'''
        fusion = fuse_mean(emg,eeg)
    elif alg=='svm':
        '''SVM fusion is not done here, just keeping system happy'''
        fusion = fuse_mean(emg,eeg)
    elif alg=='lda':
        '''LDA fusion is not done here, just keeping system happy'''
        fusion = fuse_mean(emg,eeg)
    elif alg=='rf':
        '''RF fusion is not done here, just keeping system happy'''
        fusion = fuse_mean(emg,eeg)
    else:
        msg='Fusion algorithm '+alg+' not recognised'
        raise NotImplementedError(msg)
    return fusion

def fuse_mean(mode1,mode2):
    mean=fuse_linweight(mode1,mode2,50,50)
    return mean

def fuse_single_autocorr(mode1,mode2,priorarr,lag):
    '''selects a lin weighting to maximise autocorreletion of the fused sig'''
    ind_train1 = np.argmax(mode1.sum(axis=0))
    ind_train2 = np.argmax(mode2.sum(axis=0))
    mode1=mode1[:,ind_train1]
    mode2=mode2[:,ind_train2]
    weights=np.linspace(0,1,num=110)
    poss_fuse=[]
    corrs=[]
    for i in range(len(weights)):
        w1=weights[i]
        w2=1-w1
        poss_fuse.append(fuse_linweight(mode1,mode2,w1,w2))
        corrs.append(calc_autocorr(np.append(priorarr,poss_fuse[i]),lag))
    ind=np.argmax(corrs)
    opt_fused=poss_fuse[ind]
    w1_sel=weights[ind]
    w2_sel=1-w1_sel
    return w1_sel,w2_sel

def calc_autocorr(probs,k):
    avg=np.mean(probs)
    numer=0.0
    denom=0.0
    for i in range(len(probs)-k):
        meandiff=probs[i]-k
        numer+=meandiff*(probs[i+k]-avg)
        denom+=(meandiff**2)
    autocorr=numer/denom
    return autocorr

def calc_autocorr_lag(probs,lagfactor):
    autocorrs=[]
    for i in range(lagfactor):
        autocorrs.append(abs(calc_autocorr(probs,i+1)))
    autocorr=sum(autocorrs)/lagfactor
    return autocorr

def get_w_autocorr(mode1,mode2):
    lagfactor=2
    ind_train1 = np.argmax(mode1.sum(axis=0))
    ind_train2 = np.argmax(mode2.sum(axis=0))
    autocorr_m1 = calc_autocorr_lag(mode1[:,ind_train1],lagfactor)
    autocorr_m2 = calc_autocorr_lag(mode2[:,ind_train2],lagfactor)
    w1=autocorr_m1/(autocorr_m1+autocorr_m2)
    w2=autocorr_m2/(autocorr_m1+autocorr_m2)
    print('w1: ',w1)
    print('w2: ',w2)
    return w1,w2

def get_weights(mode1,mode2,method,loud=0):
    if method=='JS':
        #w1,w2=get_initial_js(mode1,mode2,loud)
        #trialling alt js strat... sometimes trips nan?
        w1,w2=get_js_weightedprob(mode1, mode2)
    elif method=='autocorr' or method == 'autocov':
        w1,w2=get_w_autocorr(mode1, mode2)
    else:
        print('no method chosen, assigning mean')
        w1=0.5
        w2=0.5
    return w1,w2
    
def fuse_js_offline(mode1,mode2):
    wjs1,wjs2=get_initial_js(mode1,mode2)
    fused=[]
    prior=fuse_linweight(mode1,mode2,wjs1,wjs2)
    fused.append(prior)
    for instance in range(len(mode1)):
        fusedinstance,wjs1,wjs2=fuse_single(mode1,mode2,wjs1,wjs2,prior)
        fused.append(fusedinstance)
    return fused
    
def fuse_single(mode1,mode2,w1,w2,prior,loud=0):
    fused=prior*((mode1*w1) + (mode2*w2))
    #following line is uncommented for non-temporal
    #fusion(i,:) =  ( testmod1(i,:).*js_w1 ) + ( testmod2(i,:).*js_w2 );
    fused = fused/fused.sum()
    if loud:
        print('m1 dist: ',np.around(mode1,3))
        print('m2 dist: ',np.around(mode2,3))
        print('prior dist: ',np.around(prior,3))
        print('fused dist: ',np.around(fused,3))
    #temporal: it's needed to absorb transitions
    ind = np.argmax(fused);
    if (fused[0,ind] > 0.999):
        _,c = np.shape(mode1);
        fused = [0.001/(c-1)]*c;
        fused[ind] = 0.999;
    width=prior.shape[1]
    return np.reshape(np.asarray(fused),(1,width))
    '''if (fused[0,ind] == 1):
        fused[0,ind]=0.99999
    return fused'''

def fuse_js_loop(mode1,mode2,w1,w2):
    fusion = [];
    r,c = np.shape(mode1);
    prior=[1/3]*c;
    i=1
    while i<=r:
        prior = fuse_single(mode1[i,:],mode2[i,:],w1,w2,prior)
        fusion.append(prior)
        i+=1
    return fusion

def calc_entropy(probs):
    probs=np.asarray(probs)
    probsq=np.square(probs[probs!=0])
    ents=probsq[probsq!=0]*np.log(probsq[probsq!=0])
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

def get_js_weightedprob(mode1,mode2,loud=0):
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
    #weight the probs
    weighted=(mode1*w1)+(mode2*w2)
    #KL divergence P(y|w) and P(w|y)
    # modality1
    KLpyw_m1 = sum(mode1[:,ind_train1]*(mode1[:,ind_train1]/weighted[:,ind_train1]));
    KLpwy_m1 = sum(weighted[:,ind_train1]*(weighted[:,ind_train1]/mode1[:,ind_train1])); 
    # modality2
    KLpyw_m2 = sum(mode2[:,ind_train2]*(mode2[:,ind_train2]/weighted[:,ind_train1]));
    KLpwy_m2 = sum(weighted[:,ind_train1]*(weighted[:,ind_train1]/mode2[:,ind_train2]));
    #JS divergence
    JS_mod1 = (KLpyw_m1 + KLpwy_m1) * 0.5;
    JS_mod2 = (KLpyw_m2 + KLpwy_m2) * 0.5; 
    if np.isinf(JS_mod1):
        JS_mod1=1
    if np.isinf(JS_mod2):
        JS_mod2=1
    # JS weights: normalization
    #tot = (JS_mod1 + JS_mod2);
    #wjs1 = JS_mod1 / tot;
    #wjs2 = JS_mod2 / tot;
    wjs1,wjs2=normalise_weights(JS_mod1,JS_mod2);
    #should the above be 1-w, to reward Low divergence rather than High?
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

#def w_sanity_check(w):
    

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
  
if __name__=='__main__':
    if 1:
        mode1=np.asarray([[0.25, 0.25, 0.5],[0.2, 0.2, 0.6],[0, 0.3, 0.7],[0.1, 0.5, 0.4]])
        mode2=np.asarray([[0.2, 0.35, 0.45],[0.2, 0.2, 0.6],[0, 0.45, 0.55],[0.5, 0.1, 0.4]])
        #w1,w2=get_initial_js(mode1, mode2,1)
        #w1,w2=get_w_autocorr(mode1, mode2)
        fusion_mean=fuse_mean(mode1,mode2)
        fusion_max=fuse_max_arr(mode1,mode2)