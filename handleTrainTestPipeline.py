#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 01:35:12 2022

@author: pritcham
"""

import os
import numpy as np
import handleDatawrangle as wrangle
import handleFeats as feats
import handleML as ml
import handleComposeDataset as comp
import handleFusion as fus
import testFusion as testFus
import params
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay #plot_confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from random import randint


def process_data(datatype,datain=None,overwrite=True,bf_time_moved=False,dataout=None):
    '''datatype = 'emg' or 'eeg',
    datain = directory of raw unprocessed data,
    overwrite = Modify the data in-place T/F?'''
    
    root="/home/michael/Documents/Aston/MultimodalFW/"
    
    if datain is None:
        datain=askdirectory(initialdir=root,title='Select '+datatype.upper()+' Directory')
    
    if overwrite:
        dataout=datain
    else:
        if dataout is None:
            dataout=askdirectory(initialdir=root,title='Select Directory for processed '+datatype.upper())
    
    if datatype=='emg':
        wrangle.process_emg(datain,dataout)        
    elif datatype=='eeg':
        datacolsrearr=datain
        wrangle.process_eeg(datain,datacolsrearr,dataout,bf_time_moved)
    else:
        raise TypeError('Error: Unknown data type: '+str(datatype))
    
    print('**processed raw '+datatype+'**')
    #sync_raw_files()    #not yet ready as EEG data not ready
    return dataout


def make_feats(datatype,data_path):
    '''handleFeats make_feats is cleaner for a manual feature pull'''
    feats_file=asksaveasfilename(initialdir=root,title='Save featureset as')
    feats.make_feats(directory_path=data_path,output_file=feats_file,datatype=datatype)
    print('**made '+datatype+' featureset**')
    return feats_file


def split_train_test(featspath):
    featset=ml.matrix_from_csv_file(featspath)[0]
    print('**loaded featureset**')
    #split the dataset into train and test
    train,test=ml.skl.model_selection.train_test_split(featset,test_size=0.25,shuffle=False)
    print('**split train/test**')

    train_path=featspath[:-4]+'_trainslice.csv'
    test_path=featspath[:-4]+'_testslice.csv'
    
    np.savetxt(train_path, train, delimiter = ',')
    np.savetxt(test_path, test, delimiter = ',')
    print('**saved train/test splits')
    return train_path, test_path


def train(train_path):
    ml.train_offline('RF',train_path)
    print('**trained a model**')


def test(datatype,test_set_path=None):
    ''' datatype = 'emg' or 'eeg',
    test_set_path = test slice featureset '''
    
    root="/home/michael/Documents/Aston/MultimodalFW/"
    if (not 'test' in locals()) and (test_set_path is None):
        # could this just be argument test_set_path=None, and then if is None?
        test_set_path=askopenfilename(initialdir=root,title='Select test set')
    test=ml.matrix_from_csv_file_drop_ID(test_set_path)[0]
    testset_values = test[:,0:-1]
    testset_labels = test[:,-1]
    
    model = ml.load_model('testing '+str(datatype),root)
    labels=model.classes_
    
    distrolist=[]
    predlist=[]
    correctness=[]
    
    for inst_count, instance in enumerate(testset_values):
        distro=ml.prob_dist(model, instance.reshape(1,-1))  
        predlabel=ml.pred_from_distro(labels, distro)
        distrolist.append(distro)
        predlist.append(predlabel)
        if predlabel == testset_labels[inst_count]:
            correctness.append(True)
        else:
            correctness.append(False)
    accuracy = sum(correctness)/len(correctness)
    print(accuracy)
    
    gest_truth=[params.idx_to_gestures[gest] for gest in testset_labels]
    gest_pred=[params.idx_to_gestures[pred] for pred in predlist]
    gesturelabels=[params.idx_to_gestures[label] for label in labels]
    
    confmat(gest_truth,gest_pred,gesturelabels,testset=test_set_path)
    return gest_truth,distrolist,gest_pred


def confmat(y_true,y_pred,labels,modelname="",testset=""):
    '''y_true = actual classes, y_pred = predicted classes,
    labels = names of class labels'''
    conf=confusion_matrix(y_true,y_pred,labels=labels)
    cm=ConfusionMatrixDisplay(conf,labels).plot()
    cm.figure_.suptitle=(modelname+'\n'+testset)
    #add the model name and test set as labels?? using suptitle here didnt work lol
    plt.show()
    
    
def copy_files(filelist,emg_dest,eeg_dest):
    '''filelist = list of data files in master dataset directory,
    emg_dest = destination path for EMG files,
    eeg_dest = destination path for EEG files'''
    for file in filelist:
        if file[-7:-4]=='EEG':
            dest=os.path.join(eeg_dest,file)
        else:
            dest=os.path.join(emg_dest,file)
        source=os.path.join(path,file)
        if not os.path.exists(dest):
            comp.copyfile(source,dest)
            

def within_ppt_fuse(eeg_set_path=None,emg_set_path=None,single_ppt_dataset=False,selected_ppt=1,args=None):
    '''use handleFeats make_feats on a dir of data for a manual featset gen.
    selected_ppt not needed if single_ppt_dataset is True'''
    
    if args is None:
        args={'eeg_model_type':'RF','emg_model_type':'RF','fusion_alg':'mean',
                 'n_trees':20}
    if eeg_set_path is None:
        eeg_set_path='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/P4_CSVs/P4_EEG8chFeats.csv'
    eeg_set=ml.pd.read_csv(eeg_set_path,delimiter=',')
    
    if emg_set_path is None:
        emg_set_path='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/P4_CSVs/P4_EMGFeats.csv'
    emg_set=ml.pd.read_csv(emg_set_path,delimiter=',')
    
    if not single_ppt_dataset:
        eeg_masks=fus.get_ppt_split(eeg_set)
        emg_masks=fus.get_ppt_split(emg_set)
        eeg_ppt_mask=eeg_masks[selected_ppt]
        emg_ppt_mask=emg_masks[selected_ppt]
        eeg_ppt = eeg_set[eeg_ppt_mask]
        emg_ppt=emg_set[emg_ppt_mask]
    else:
        eeg_ppt=eeg_set
        emg_ppt=emg_set
    
    #eeg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    #index_eeg=ml.pd.MultiIndex.from_arrays([eeg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    eeg_ppt['ID_stratID']=eeg_ppt['ID_run']+eeg_ppt['Label']+eeg_ppt['ID_gestrep']
    emg_ppt['ID_stratID']=emg_ppt['ID_run']+emg_ppt['Label']+emg_ppt['ID_gestrep']
    random_split=randint(0,100)
    train_split,test_split=train_test_split(eeg_ppt['ID_stratID'].unique(),test_size=0.33,random_state=random_split)
    
    eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(train_split)]
    eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split)]
    eeg_train=ml.drop_ID_cols(eeg_train)
    
    emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(train_split)]
    emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split)]
    emg_train=ml.drop_ID_cols(emg_train)
    
    
    
    eeg_model_type=args['eeg_model_type']
    eeg_model = ml.train_optimise(eeg_train, eeg_model_type, args)
    emg_model_type=args['emg_model_type']
    emg_model = ml.train_optimise(emg_train, emg_model_type, args)
    classlabels = eeg_model.classes_
    
    eeg_test_set=ml.drop_ID_cols(eeg_test)
    eeg_test_labels=eeg_test_set['Label'].values
    eeg_test_set=eeg_test_set.drop(['Label'],axis=1)
    preds=eeg_model.predict(eeg_test_set)
    eeg_acc=accuracy_score(eeg_test_labels,preds)
    print('EEG accuracy: '+str(eeg_acc))
    confmat(eeg_test_labels,preds,classlabels)
    
    emg_test_set=ml.drop_ID_cols(emg_test)
    emg_test_labels=emg_test_set['Label'].values
    emg_test_set=emg_test_set.drop(['Label'],axis=1)
    preds=emg_model.predict(emg_test_set)
    emg_acc=accuracy_score(emg_test_labels,preds)
    print('EMG accuracy: '+str(emg_acc))
    confmat(emg_test_labels,preds,classlabels)
    
    
    '''fusion.refactor_synced drops ID cols so have to pass the one with the IDcols'''
    targets, _, _, predlist_fusion = testFus.refactor_synced_predict(emg_test,eeg_test,emg_model,eeg_model,classlabels,args)
    gest_truth=[params.idx_to_gestures[gest] for gest in targets]
    gest_pred_fusion=[params.idx_to_gestures[pred] for pred in predlist_fusion]
    gesturelabels=[params.idx_to_gestures[label] for label in classlabels]
    
    fusion_acc=accuracy_score(gest_truth,gest_pred_fusion)
    print('Fusion accuracy: '+str(fusion_acc))
    confmat(gest_truth,gest_pred_fusion,gesturelabels)

            
def within_ppt_test(set_path=None,single_ppt_dataset=False,selected_ppt=1,args=None,datatype=''):
    '''use handleFeats make_feats on a dir of data for a manual featset gen.
    selected_ppt not needed if single_ppt_dataset is True'''
    
    if args is None:
        args={'model_type':'RF',
                     'n_trees':20}
    if set_path is None:
        set_path='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EEG/featsEEGNewDecImpulseKill.csv'
    dataset=ml.pd.read_csv(set_path,delimiter=',')
    
    if not single_ppt_dataset:
        ppt_masks=fus.get_ppt_split(dataset)
        ppt=ppt_masks[selected_ppt]
        data_ppt = dataset[ppt]
    else:
        data_ppt=dataset
    
    #eeg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    #index_eeg=ml.pd.MultiIndex.from_arrays([eeg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    data_ppt['ID_stratID']=data_ppt['ID_run']+data_ppt['Label']+data_ppt['ID_gestrep']
    train_split,test_split=train_test_split(data_ppt['ID_stratID'].unique(),test_size=0.33)
    data_train=data_ppt[data_ppt['ID_stratID'].isin(train_split)]
    data_test=data_ppt[data_ppt['ID_stratID'].isin(test_split)]
    data_train=ml.drop_ID_cols(data_train)
    
    model_type=args['model_type']
    model = ml.train_optimise(data_train, model_type, args)
    classlabels = model.classes_
    
    test_set=ml.drop_ID_cols(data_test)
    test_labels=test_set['Label'].values
    test_set=test_set.drop(['Label'],axis=1)
    preds=model.predict(test_set)
    acc=accuracy_score(test_labels,preds)
    print(datatype+'accuracy: '+str(acc))
    confmat(test_labels,preds,classlabels)


if __name__ == '__main__':
    
    eeg_wayg_set='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/P4_CSVs/P4_EEG8chFeats.csv'
    emg_wayg_set='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/P4_CSVs/P4_EMGFeats.csv'
    
    eeg_wayg_set='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/P8_CSVs/P8_EEGFeats.csv'
    emg_wayg_set='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/P8_CSVs/P8_EMGFeats.csv'
    
    eeg_wayg_set='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/P8_CSVs/P8_EEGFeats.csv'
    emg_wayg_set='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/P8_CSVs/P8_EMGFeats.csv'
    
    
    #within_ppt_test(eeg_wayg_set,single_ppt_dataset=True,datatype='EEG')
    #within_ppt_test(emg_wayg_set,single_ppt_dataset=True,datatype='EMG')
    
    within_ppt_fuse(eeg_wayg_set,emg_wayg_set,single_ppt_dataset=True)
    raise
    #run handleComposeDataset
    #need some way of doing leave-ppt-out crosseval.
    #maybe just n runs of ComposeDataset but skipping the gui?
    #OR: just get feature files for each ppt and then assemble as needed?
    
    #need also to make train test split not random but split BY TRIALS!
    #so probably do a stratified split of the raw datafiles first?
    #sklearn train_test_split stratified on the list of files?
    
    #here=os.path.dirname(os.path.realpath(__file__))
    
    root="/home/michael/Documents/Aston/MultimodalFW/"
    working = root + 'working_dataset/'
    path_devset=root+'dataset/dev/'
    
    pptlist = ['1 - M 24','2 - M 42','4 - F 36',
               '7 - F 29','8 - F 27','9 - F 24',
               '11 - M 24','13 - M 31','14 - M 28']
    paths=[]
    for ppt in pptlist:
        paths.append(comp.build_path(path_devset,ppt.split(' ')[0]))
    
    trainsize=0.75
    for path in paths:
        files=os.listdir(path)
        pptnum=path.split('/')[-1]
        labels=[file.split('-')[1] for file in files]
        labelled_files=(np.asarray([files,labels])).transpose()
        trainfiles,testfiles=ml.skl.model_selection.train_test_split(labelled_files,train_size=trainsize,stratify=labels)
        trainfiles=trainfiles[:,0].tolist()
        testfiles=testfiles[:,0].tolist()
        
        train_emg=working+str(pptnum)+'/trainsplit/EMG/'
        train_eeg=working+str(pptnum)+'/trainsplit/EEG/'
        comp.make_path(train_emg)
        comp.make_path(train_eeg)
            
        test_emg=working+str(pptnum)+'/testsplit/EMG/'
        test_eeg=working+str(pptnum)+'/testsplit/EEG/'
        comp.make_path(test_emg)
        comp.make_path(test_eeg)
        
        copy_files(trainfiles,train_emg,train_eeg)
        copy_files(testfiles,test_emg,test_eeg)
        
        train_emg=process_data(train_emg)
        test_emg=process_data(test_emg)
        
        train_emg_featset=working+str(pptnum)+'_EMG_train.csv'
        test_emg_featset=working+str(pptnum)+'_EMG_test.csv'
        
        feats.make_feats(train_emg,train_emg_featset,'emg')
        feats.make_feats(test_emg,test_emg_featset,'emg')
        
        train(train_emg_featset)
        true,distros,preds=test(test_emg_featset)
        
        break
    
    
    raise #below is just for a one and done not stratifying
    print('stop')
    Tk().withdraw()
    
    processed_emg_path=process_data()
    emg_feats_filepath=make_feats(processed_emg_path)
    train_set_path, test_set_path = split_train_test(emg_feats_filepath)
    train(train_set_path)
    test(test_set_path=None)
    
    

