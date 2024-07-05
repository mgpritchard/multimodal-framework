#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 23:42:00 2022

@author: pritcham
"""

import os
import sys
import numpy as np
import statistics as stats
#import handleDatawrangle as wrangle
#import handleFeats as feats
import handleML as ml
#import handleComposeDataset as comp
#import handleTrainTestPipeline as tt
#import handleFusion as fusion
import params
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, log_loss, confusion_matrix, ConfusionMatrixDisplay #plot_confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import random
import matplotlib.pyplot as plt
#from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, Trials
#from hyperopt.pyll import scope, stochastic
import time
import pandas as pd
import pickle as pickle

import tensorflow as tf



def balance_single_mode(dataset):
    dataset['ID_stratID']=dataset['ID_pptID'].astype(str)+dataset['Label'].astype(str)
    stratsize=np.min(dataset['ID_stratID'].value_counts())
    balanced = dataset.groupby('ID_stratID')
    #g.apply(lambda x: x.sample(g.size().min()))
    #https://stackoverflow.com/questions/45839316/pandas-balancing-data
    balanced=balanced.apply(lambda x: x.sample(stratsize))
    print('subsampling to ',str(stratsize),' per combo of ppt and class')
    return balanced

def confmat(y_true,y_pred,labels,modelname="",testset="",title=""):
    '''y_true = actual classes, y_pred = predicted classes,
    labels = names of class labels'''
    #https://scikit-learn.org/0.22/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
    #https://scikit-learn.org/0.22/modules/generated/sklearn.metrics.plot_confusion_matrix.html#sklearn.metrics.plot_confusion_matrix
    conf=confusion_matrix(y_true,y_pred,labels=labels,normalize='true')
    cm=ConfusionMatrixDisplay(conf,labels)
    #cm=ConfusionMatrixDisplay.from_predictions(y_true,y_pred,labels,normalise=None) #only in skl 1.2
    if modelname != "" and testset != "":
        title=modelname+'\n'+testset
    fig,ax=plt.subplots()
    ax.set_title(title)
    cm.plot(ax=ax)
    plt.show()
    #return conf

def get_ppt_split_flexi(featset):
    masks=[featset['ID_pptID']== n_ppt for n_ppt in np.sort(featset['ID_pptID'].unique())] 
    return masks



def classes_from_preds(targets,predlist_emg,classlabels):
    '''Convert predictions to gesture labels'''
    gest_truth=[params.idx_to_gestures[gest] for gest in targets]
    gest_pred_emg=[params.idx_to_gestures[pred] for pred in predlist_emg]
    gesturelabels=[params.idx_to_gestures[label] for label in classlabels]
    
    return gest_truth,gest_pred_emg,gesturelabels

def plot_confmats(gest_truth,gest_pred_emg,gesturelabels):
        '''Produce confusion matrix'''
        confmat(gest_truth,gest_pred_emg,gesturelabels)
    #CAN you have a consistent gradation of the colour heatmap across confmats?
    #ie yellow is always a fixed % not relative to the highest in that given
    #confmat

def train_models_opt(emg_train_set,args):
    emg_model_type=args['emg']['emg_model_type']
    emg_model = ml.train_optimise(emg_train_set, emg_model_type, args['emg'])
    return emg_model






###########################################

def get_fresh_model():
    # https://github.com/stelehm/Deep-transfer-learning-compared-to-subject-specific-models-for-sEMG-decoders/blob/main/3%20Experiments/experiment_DB3_raw.py
    
    #from optuna:
    #batch-size = 512
    # beta1 = 0.129 - 0.214
    # beta2 = 0.805 - 0.9722
    # epsilon = 0.000026 - 0.000118
    # learning_rate = 0.00012 - 0.000257
    
    # cnn_dropout = 0.0105 - 0.103
    # cnn_kernel_size = 3
    # cnn_input_nb_kernel = 19 -44
    # cnn_nb_layers = 2
    # cnn_hidden_nb_kernel_0 = 19-47
    # cnn_hidden_nb_kernel_1 = 40 - 66
    # dense_dropout = 0.158 -0.232
    # dense_nb_layers = 1
    # dense_hidden_0 = 92 - 128
    
    nb_cnn_input_kernel = 32
    cnn_kernel_size = 3
    cnn_layers = [32, 64]
    cnn_dropout_factor = 0.01
    dense_layers = [128]
    dense_dropout_factor = 0.1
    # above values in their github repo, and in https://iopscience.iop.org/article/10.1088/1741-2552/ac9860/meta Table 3
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(nb_cnn_input_kernel, cnn_kernel_size, padding='same', activation='relu', kernel_regularizer="l2", 
                                     input_shape=(150,12)))
    # theirs is shape=(400,12). They have 12 electrodes, and 200ms windows at 2kHz = 400 samples
    # I have 6 electrodes + 6 lag windows = 12 channels, and 150 samples from 500ms data (since feature script resamples to 150)
    # so my shape is (150, 12)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D())
    model.add(tf.keras.layers.Dropout(cnn_dropout_factor))
    for layer_size in cnn_layers:
        model.add(tf.keras.layers.Conv1D(layer_size, cnn_kernel_size, padding='same', activation='relu', kernel_regularizer="l2"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D())
        model.add(tf.keras.layers.Dropout(cnn_dropout_factor))
    model.add(tf.keras.layers.Flatten())
    for layer_size in dense_layers:
        model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
        model.add(tf.keras.layers.Dropout(dense_dropout_factor))
    model.add(tf.keras.layers.Dense(4, kernel_regularizer="l2", activation="softmax"))
    # final dense layer is 1 node per class, theirs is 17 for ninapro DB2/3, mine is 4
    return model


def build_model():
    #build model
    pretrained_model = get_fresh_model()
    learning_rate = 0.0002
    beta_1 = 0.2
    beta_2 = 0.9
    epsilon = 0.0001
    
    ##opt = tf.keras.optimizers.Adam(learning_rate = learning_rate, beta_1=beta_1, beta_2 = beta_2,epsilon=epsilon)
    #opt = tf.keras.optimizers.Adam()
    # their code in github repo just uses Adam() with no params. defaults are learning=0.001 b1=0.9 b2=0.999 eps = 1e-7
    # as per https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
    # but their paper Table 3 shows the listed values above, so i will use those
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate, beta_1=beta_1, beta_2 = beta_2,epsilon=epsilon)

    pretrained_model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    
    return pretrained_model
    
    


def get_categorical(y):
    return pd.get_dummies(pd.Series(y)).values


def deep_bespoke(args):
    start=time.time()
    if not args['data_in_memory']:
        emg_set_path=args['emg_set_path']
        emg_set=ml.pd.read_pickle(emg_set_path)
    else:
        emg_set=args['emg_set']
        
    emg_masks=get_ppt_split_flexi(emg_set)
    
    for idx,emg_mask in enumerate(emg_masks):
        
        emg_ppt = emg_set[emg_mask]
        
        emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        emg_ppt['ID_stratID']=emg_ppt['ID_run'].astype(str)+emg_ppt['Label'].astype(str)+emg_ppt['ID_gestrep'].astype(str)
        
        random_split=random.randint(0,100)
        
        gest_perfs=emg_ppt['ID_stratID'].unique()
        gest_strat=pd.DataFrame([gest_perfs,[''.join(filter(str.isalpha, perf)) for perf in gest_perfs]]).transpose()
        train_split,test_split=train_test_split(gest_strat,test_size=0.33,random_state=random_split,stratify=gest_strat[1])

        emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(train_split[0])]
        emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]
        
        emg_train=ml.drop_ID_cols(emg_train)
        emg_test=ml.drop_ID_cols(emg_test)
        
        y_train=emg_train.pop('Label')
        X_train=emg_train
        
        y_train=y_train.to_numpy()
        X_train=X_train.to_numpy()
        #shuffle
        random_ids_train = list(range(y_train.shape[0]))
        random.shuffle(random_ids_train)
        X_train= X_train[random_ids_train]
        y_train= y_train[random_ids_train]
        
        X_train=np.swapaxes(np.vstack([np.dstack(X_train[:,n]) for n in range(12)]),0,2)
        # going from 2d array of lists to 3d array
        
        y_train = get_categorical(y_train)
        # categorical_crossentropy requires a onehot encoding
        
        deep_model=build_model()
        
        
        # their code monitors val_accuracy, but my TF version does not match names to the declared 'accuracy' metric
        # and instead creates a metric called val_acc. if i try to monitor val_accuracy, it cannot.
        # based on their stated TF version, i *believe* their code did indeed stop early, and so mine should too (also because its their stated method in the paper).
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', 
                                                      patience=10, 
                                                      min_delta=0.001,
                                                      mode='auto',
                                                      restore_best_weights=True
                                                      )
        batchsize = 512
        # both as per their github repo and paper section 3.3.4
        
        epochs = 150
        #epochs = 15 #temp, for code testing
        
        history = deep_model.fit(X_train, y_train, epochs=epochs ,validation_split=0.1,batch_size=batchsize, callbacks=[early_stopping])
        
        traintime = time.time()
        
        train_loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]
        train_accuracy = history.history["acc"][-1]
        # these need to be 'acc' and 'val_acc' not 'accuracy'. possibly related to https://www.codesofinterest.com/2020/01/fixing-keyerror-acc-valacc-keras.html
        val_accuracy = history.history["val_acc"][-1]
    
        nb_train_epochs = len(history.history['loss'])
        
        
        
        y_test=emg_test.pop('Label')
        X_test=emg_test
        y_test=y_test.to_numpy()
        X_test=X_test.to_numpy()
        
        X_test=np.swapaxes(np.vstack([np.dstack(X_test[:,n]) for n in range(12)]),0,2)
        y_test = get_categorical(y_test)
        
        test_accuracy = deep_model.evaluate(X_test, y_test)[1]
        
        end = time.time()
        
        subject=str(emg_ppt['ID_pptID'].iloc[0])
        if args['plot_confmats']:
            y_pred = deep_model.predict(X_test)
            y_pred = np.argmax (y_pred, axis = 1)
            y_pred = [params.idx_to_gestures_deeplearn[pred] for pred in y_pred]
            y_true = np.argmax(y_test, axis=1)
            y_true = [params.idx_to_gestures_deeplearn[targ] for targ in y_true]
            #Create confusion matrix and normalizes it over predicted (columns)
            #result = confusion_matrix(y_testLabs, y_predLabs , normalize='pred')
            labels=['Cyl','Lat','Rest','Sph']
            labels=[0,1,2,3]
            labels=['cylindrical','spherical','lumbrical','rest']
            confmat(y_true,y_pred,labels,modelname="",testset="",title="")     
        
        
        resultsdict={
            'subject':subject,
            'train_epochs':nb_train_epochs,
            'train_loss':train_loss,
            'val_loss':val_loss,
            'train_accuracy':train_accuracy,
            'val_accuracy':val_accuracy,
            'test_accuracy':test_accuracy,
            'elapsed_time':end-start,
            'training_time':traintime-start,
            }
        
        
        return resultsdict, deep_model, history.history
        


def function_fuse_withinppt(args):
    start=time.time()
    if not args['data_in_memory']:
        emg_set_path=args['emg_set_path']
    
        emg_set=ml.pd.read_csv(emg_set_path,delimiter=',')
    else:
        emg_set=args['emg_set']

    emg_masks=get_ppt_split_flexi(emg_set)
    
    emg_accs=[] #https://stackoverflow.com/questions/13520876/how-can-i-make-multiple-empty-lists-in-python   
    train_accs=[]

    for idx,emg_mask in enumerate(emg_masks):
        
        emg_ppt = emg_set[emg_mask]
        #emg_others = emg_set[~emg_mask]

        emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        emg_ppt['ID_stratID']=emg_ppt['ID_run'].astype(str)+emg_ppt['Label'].astype(str)+emg_ppt['ID_gestrep'].astype(str)
        

        random_split=random.randint(0,100)

        gest_perfs=emg_ppt['ID_stratID'].unique()
        gest_strat=pd.DataFrame([gest_perfs,[perf.split('.')[1][-1] for perf in gest_perfs]]).transpose()
        train_split,test_split=train_test_split(gest_strat,test_size=0.33,random_state=random_split,stratify=gest_strat[1])

        emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(train_split[0])]
        emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]

     
        if args['fusion_alg']=='just_emg':
            
            if not args['get_train_acc']:
                targets, predlist_emg, classlabels=only_EMG(emg_train, emg_test, args)
            else:
                targets, predlist_emg, classlabels, traintargs, predlist_train=only_EMG(emg_train, emg_test, args)
        
        else:
            
            if args['get_train_acc']:
                emg_trainacc=emg_train.copy()
                emg_trainacc.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
           
            emg_train=ml.drop_ID_cols(emg_train)           
            
            emg_model=train_models_opt(emg_train,args)
        
            classlabels = emg_model.classes_
            
            emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
                
            targets, predlist_emg,_ = refactor_synced_predict(emg_test, emg_model, classlabels,args)

            if args['get_train_acc']:
                traintargs, predlist_emgtrain,_ = refactor_synced_predict(emg_trainacc, emg_model, classlabels, args)

        
        gest_truth,gest_pred_emg,gesturelabels=classes_from_preds(targets,predlist_emg,classlabels)

        if args['plot_confmats']:
            gesturelabels=[params.idx_to_gestures[label] for label in classlabels]
            confmat(gest_truth,gest_pred_emg,gesturelabels,title='EMG')
            
        emg_accs.append(accuracy_score(gest_truth,gest_pred_emg))

        
        if args['get_train_acc']:
            train_truth=[params.idx_to_gestures[gest] for gest in traintargs]
            train_preds=[params.idx_to_gestures[pred] for pred in predlist_train]
            train_accs.append(accuracy_score(train_truth,train_preds))
        else:
            train_accs.append(0)
        
    mean_acc=stats.mean(emg_accs)
    mean_emg=stats.mean(emg_accs)
    median_emg=stats.median(emg_accs)

    mean_train_acc=stats.mean(train_accs)
    end=time.time()
    #return 1-mean_acc
    return {
        'loss': 1-mean_acc,
        'status': STATUS_OK,
        'emg_mean_acc':mean_emg,
        'emg_median_acc':median_emg,
        'emg_accs':emg_accs,
        'mean_train_acc':mean_train_acc,
        'elapsed_time':end-start,}



        

    trainEMGpath=r"H:\Jeong11tasks_data\deepLcompare\final_set\noHoldout_RawEMG.pkl"
    
    
def test_deep_once_devppt():
    trial_set_path=r"H:\Jeong11tasks_data\deepLcompare\final_set\dev_ppt_4_RawEMG.pkl"
        
    args={'emg_set_path':trial_set_path,
          'data_in_memory':False,
          'plot_confmats':True}
    
    resultsdict, deep_model, history = deep_bespoke(args)
    

    #write results
    subject=resultsdict['subject']
    foldername=r"H:\Jeong11tasks_data\deepLcompare/"
    deep_model.save_weights(foldername+"model_ppt{}".format(subject))
    
    with open(foldername+"results_ppt{}.csv".format(subject),"w") as wf:
         #wf.write("subject,train_epochs,train_loss,val_loss,train_accuracy,val_accuracy,test_accuracy,elapsed_time,training_time")
         wf.write(','.join(list(resultsdict.keys())))
         wf.write("\n")
    
    #variable_list = [subject,nb_train_epochs,train_loss,val_loss,train_accuracy,val_accuracy,test_accuracy,end-start,traintime-start]
    with open(foldername+"results_ppt{}.csv".format(subject),"a") as wf:
        #wf.write(', '.join([str(measure) for measure in variable_list ]))
        wf.write(', '.join([str(measure) for measure in resultsdict.values() ]))
        wf.write("\n")
        
    with open(foldername+"history_ppt{}.pkl".format(subject),"wb") as logfile:
        pickle.dump(history,logfile)
    
    logs=pickle.load(open(r"H:\Jeong11tasks_data\deepLcompare\history_ppt4.pkl",'rb'))
    plt.plot(logs['acc'])
    plt.plot(logs['val_acc'])


    
if __name__ == '__main__':
    

    ppt_scores = deep_holdouts()
    
    ppt_scores.reset_index(drop=False)
    ppt_scores.to_csv(r"H:\Jeong11tasks_data\deepLcompare\bespoke_litDeep_100_repeats.csv")
    
    
  