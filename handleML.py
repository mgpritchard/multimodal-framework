#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 22:37:59 2021

@author: pritcham

module to contain functionality related to ML classification
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import sklearn as skl
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale, label_binarize
import types

#from hyperopt import fmin, tpe, hp, STATUS_OK
#from hyperopt.pll import scope

import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename
import time
'''look into adding logistic regression classifier?
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'''

def eval_acc(results,state):
    count=results.count(state)
    acc=count/len(results)
    return acc

def eval_fusion(mode1,mode2,fusion,state):
    acc1=eval_acc(mode1,state)
    acc2=eval_acc(mode2,state)
    accF=eval_acc(fusion,state)
    return [acc1,acc2,accF]

def eval_multi(results,state):
    acc=[]
    for result in results:
        acc.append(eval_acc(result,state))
    return acc

def load_model(name,path):
    title='select saved model for '+name
    Tk().withdraw()
    model_name=askopenfilename(initialdir=path,title=title,filetypes = (("sav files","*.sav"),("all files","*.*")))
    with open(model_name,'rb') as load_model:
        model = pickle.load(load_model)
    return model

def matrix_from_csv_file(file):
    csv_data=pd.read_csv(file,delimiter=",").values
    matrix = csv_data[1:]
    headers = csv_data[0]
    print ('MAT', (matrix.shape))
	#print ('HDR', (headers.shape))
    return matrix, headers

def drop_ID_cols(csv_dframe):
    IDs=csv_dframe.filter(regex='^ID_').columns
    csv_dframe=csv_dframe.drop(IDs,axis='columns')
    '''may benefit from a trycatch in case of keyerror?'''
    return csv_dframe

def matrix_from_csv_file_drop_ID(file):
    csv_dframe=pd.read_csv(file,delimiter=",")
    IDs=csv_dframe.filter(regex='^ID_').columns
    csv_dframe=csv_dframe.drop(IDs,axis='columns')
    #skip above 2 lines and call csv_dframe=drop_ID_cols(csv_dframe)
    matrix=csv_dframe.values
    headers = csv_dframe.columns.values
    print ('MAT', (matrix.shape))
	#print ('HDR', (headers.shape))
    return matrix, headers

def train_offline(modeltype='gaussNB',train_set=None,loaded_trainset=None,model_name=None,modeldest=None):
    path=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    Tk().withdraw()
    if loaded_trainset is None:
        if train_set is None:
            title='select training dataset'
            train_set=askopenfilename(initialdir=path,title=title,filetypes = (("csv files","*.csv"),("all files","*.*")))
        train_dat=matrix_from_csv_file_drop_ID(train_set)[0]
    else:
        train_dat=loaded_trainset
    if modeldest is None:
        title_sav='location for trained model'
        modeldest=askdirectory(initialdir=path,title=title_sav)
    if train_set is not None:
        modelname=modeldest+'/'+os.path.basename(train_set)[:-4]+'_'+modeltype+'.sav'
    else:
        modelname=modeldest+'/'+model_name+'_'+modeltype+'.csv'
    if modeltype=='gaussNB':
        model=train_nb(train_dat,modelname)
    elif modeltype=='RF':
        model=train_rf(train_dat,modelname)
    elif modeltype=='LDA':
        model=train_lda(train_dat,modelname)
    
    return model

def train_nb(train_dat,model_path):
    #data=matrix_from_csv_file(train_dat)[0] #move this up a level or 2
    naivb=GaussianNB()
    naivb,acc=train_model(naivb,train_dat)
    print('accuracy: ',acc)
    with open(model_path,'wb') as savepath:
        pickle.dump(naivb,savepath)
    return naivb

def train_rf(train_dat,model_path):
    '''internal basic optimisation over defined n_trees and saving of winner model'''
    
    #data=matrix_from_csv_file(train_dat)[0]
    randfs = dict()
	# define number of trees to consider
    n_trees = [10, 50]#, 100, 500, 1000]
    results=[]
    randfs=[]
    for n in n_trees:
        randf = RandomForestClassifier(n_estimators=n)
        randf,acc=train_model(randf,train_dat)
        print('# trees: ',n,'\naccuracy: ',acc)
        randfs.append(randf)
        results.append(acc)
    winInd=np.argmax(results)
    winner=randf[winInd]
    model_path=model_path[:-4]+'_'+str(n_trees[winInd])+'trees.sav'
    with open(model_path,'wb') as savepath:
        pickle.dump(winner,savepath)
    return winner

def train_optimise(training_set,modeltype,args):
    '''where training_set is a Pandas dataframe
    which has Label as the last column but has had ID columns dropped'''
    
    if modeltype=='RF':
        model=train_RF_param(training_set,args)
    elif modeltype=='gaussNB':
        #raise ValueError('No Gaussian NB implemented yet')
        '''Naive Bayes Reading'''
        # Bayes Net vs NB:
            #https://stackoverflow.com/questions/12298150/what-is-the-difference-between-a-bayesian-network-and-a-naive-bayes-classifier#:~:text=Bayesian%20Network%20is%20more%20complicated,some%20crucial%20attributes%20are%20discarded.
        # "it is known to be a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously"
            #https://scikit-learn.org/stable/modules/naive_bayes.html
        # Interpreting predict_proba (multinomialNB):
            #https://stackoverflow.com/questions/60166093/interpreting-predict-proba-multinomial-naive-bayes
        # "NB are bad estimators?" "LogitReg may be viable"
            #https://stackoverflow.com/questions/68080288/naive-bayes-classifiers-are-bad-estimators
        #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        
        model=train_gnb(training_set,args)
    elif modeltype=='LDA':
        #raise ValueError('No LDA implemented yet')
        model=train_LDA_param(training_set,args)
    elif modeltype=='kNN':
        model = train_knn(training_set,args)
    elif modeltype=='SVM':
        #raise ValueError('LinearSVC has no predict_proba')
        model = train_svm(training_set,args)
    elif modeltype=='QDA':
        model = train_QDA(training_set,args)
    elif modeltype=='SVM_PlattScale':
        model = train_SVC_Platt(training_set,args)
        
    # Deep model??
    #https://scikit-learn.org/stable/modules/neural_networks_supervised.html#mlp-tips
    
    #https://github.com/skorch-dev/skorch
   
    return model

def train_gnb(train_data,args):
    #data=matrix_from_csv_file(train_dat)[0] #move this up a level or 2
    smoothing=args['smoothing']
    model=GaussianNB(var_smoothing=smoothing)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model

def train_SVC_Platt(train_data,args):
    kernel=args['kernel']
    C=args['svm_C']
    gamma=args['gamma']
    if kernel=='linear':
        model=SVC(C=C,kernel=kernel,probability=True) #possible need to fix random_state as predict is called multiple times?
    else:
        model=SVC(C=C,kernel=kernel,gamma=gamma,probability=True)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model

def train_QDA(train_data,args):
    reg=args['regularisation']
    model=QuadraticDiscriminantAnalysis(reg_param=reg)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model

def train_svm(train_data,args):
    '''maybe need https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python'''
    '''or could use distro=model.decision_function(values) #https://stackoverflow.com/questions/49507066/predict-probabilities-using-svm
        distro=minmax_scale(distro,feature_range=(0,1))?'''
    C=args['svm_C']
    model=LinearSVC(C=C)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    
    def predict_proba(self, values):
    # or https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC.decision_function
        pred=self.predict(values)
        classes=self.classes_
        probs=label_binarize(pred,classes)
        #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.label_binarize.html#sklearn.preprocessing.label_binarize
        return probs
    
    model.predict_proba=types.MethodType(predict_proba,model)
    #https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance-in-python
    
    return model

def train_LDA_param(train_data,args):
    solver=args['LDA_solver']
    shrinkage=args['shrinkage']
    if solver == 'svd':
        model=LinearDiscriminantAnalysis(solver=solver)
    else:
        model=LinearDiscriminantAnalysis(solver=solver,shrinkage=shrinkage)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model

def train_knn(train_data,args):
    k=args['knn_k']
    model=KNeighborsClassifier(n_neighbors=k)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model

def train_RF_param(train_data,args):
    '''where args is a dictionary with n_trees as an integer item within'''
    n_trees=args['n_trees']
    max_depth=args['max_depth']
    model=RandomForestClassifier(n_estimators=n_trees,max_depth=max_depth)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model

def train_lda(train_dat,model_path):
    #data=matrix_from_csv_file(train_dat)[0]
    lda=LinearDiscriminantAnalysis()
    lda,acc=train_model(lda,train_dat)
    print('accuracy: ',acc)
    with open(model_path,'wb') as savepath:
        pickle.dump(lda,savepath)
    return lda
    
def train_model(model,data):
    train=data[:,:-1]
    targets=data[:,-1]
    train1,train2,test1=np.array_split(train,3)
    train1=train1.astype(np.float64)
    train2=train2.astype(np.float64)
    traindat=np.concatenate((train1,train2))
    test1=test1.astype(np.float64)
    targets1,targets2,targetstest=np.array_split(targets,3)
    targetsdat=np.concatenate((targets1,targets2))
    model.fit(traindat,targetsdat)
    results=model.predict(test1)
    acc=accuracy_score(targetstest,results)
    model.fit(train.astype(np.float64),targets)
    return model, acc

def prob_dist(model,values):
    distro = model.predict_proba(values)
    distro[distro==0]=0.00001
    return distro
#https://github.com/scikit-learn-contrib/forest-confidence-interval#readme
#for RF prediction? maybe even get conf val for all models?

def predict_from_array(model,values):
	prediction = model.predict(values)
	return prediction

def pred_from_distro(labels,distro):
    pred=int(np.argmax(distro))
    label=labels[pred]
    return label

def predlist_from_distrosarr(labels,distros):
    predcols=np.argmax(distros,axis=1)
    predlabels=labels[predcols]
    return predlabels.tolist()

def pred_gesture(prediction,toggle_print):

    if isinstance(prediction,int):
        if prediction == 2: #typically 0
            gesture='open'
        elif prediction == 1:
            gesture='neutral'
        elif prediction == 0: #typically 2
            gesture='close'
    else:
        gesture=prediction

    '''
    if prediction == 0:
        gesture='open'
    elif prediction == 1:
        gesture='neutral'
    elif prediction == 2:
        gesture='close'
    '''
        
    if toggle_print:
        print(time.time())
        print(gesture)
        print('-------------')

    return gesture  

def classify_instance(frame,model):
    prediction=frame
    return prediction

def classify_continuous(data):
    while True:
        pred=classify_instance(data,'NB')
        yield pred