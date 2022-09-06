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
import params
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #plot_confusion_matrix
import matplotlib.pyplot as plt



def process_eeg(datain=None):
    if datain is None:
        datain=askdirectory(initialdir=root,title='Select EEG Directory')
    dataout=datain
    datacolsrearr=datain
    wrangle.process_eeg(datain,datacolsrearr,dataout)
    #sync_raw_files()    #not yet ready as EEG data not ready
    print('**processed raw eeg**')
    return dataout

def make_eeg_feats(data_dir_path):
    eeg_data_path=data_dir_path
    eeg_feats_file=asksaveasfilename(initialdir=root,title='Save featureset as')
    feats.make_feats(directory_path=eeg_data_path,output_file=eeg_feats_file,datatype='eeg')
    print('**made featureset**')
    return eeg_feats_file

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

def test(test_set_path=None):
    root="/home/michael/Documents/Aston/MultimodalFW/"
    if (not 'test' in locals()) and (test_set_path is None):
        # could this just be set test as a param=None, and then if is None?
        testset_loc=askopenfilename(initialdir=root,title='Select test set')
        test=ml.matrix_from_csv_file_drop_ID(testset_loc)[0]
    else:
        test=ml.matrix_from_csv_file_drop_ID(test_set_path)[0]
        
    testset_values = test[:,0:-1]
    testset_labels = test[:,-1]
    model = ml.load_model('testing eeg',root)
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
    confmat(gest_truth,gest_pred,gesturelabels)
    return gest_truth,distrolist,gest_pred

def confmat(y_true,y_pred,labels):
    conf=confusion_matrix(y_true,y_pred,labels=labels)
    ConfusionMatrixDisplay(conf,labels).plot()
    plt.show()
    
def copy_files(filelist,emg_dest,eeg_dest):
    for file in filelist:
        if file[-7:-4]=='EEG':
            dest=os.path.join(eeg_dest,file)
        else:
            dest=os.path.join(emg_dest,file)
        source=os.path.join(path,file)
        if not os.path.exists(dest):
            comp.copyfile(source,dest)

def ditch_EEG_suffix(eegdir):
    for file in os.listdir(eegdir):
        if file.endswith('_EEG',0,-4):
            os.remove(os.path.join(eegdir,file))


## Testing the suspiciously hihgh accuracy:
#load model with testset,testset_attribs=ml.matrixdropID(testsetfile.csv)
#attrib_names = list(testset_attribs)
#plt.figure()
#tree.plot_tree(model,feature_names=attrib_names,max_depth=2,fontsize=6)    

if __name__ == '__main__':
    #test(None)
    #raise
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
    for path in paths[3:]:
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
        
        '''ONLY DO THIS IF NOT ALREADY PROCESSED EEG'''
        if 1:   
            train_eeg=process_eeg(train_eeg)
            ditch_EEG_suffix(train_eeg)
            test_eeg=process_eeg(test_eeg)
            ditch_EEG_suffix(test_eeg)
        
        train_eeg_featset=working+str(pptnum)+'_eeg_train.csv'
        test_eeg_featset=working+str(pptnum)+'_eeg_test.csv'
        
        feats.make_feats(train_eeg,train_eeg_featset,'eeg',period=1)
        feats.make_feats(test_eeg,test_eeg_featset,'eeg',period=1)
        
        #eegtrain_labelled=train_eeg_featset[:-4] + '_Labelled.csv'
        #eegtest_labelled=test_eeg_featset[:-4] + '_Labelled.csv'
        train(train_eeg_featset)
        y_true,y_distro,y_pred=test(test_eeg_featset)
        #conf=confusion_matrix(y_true,y_pred)
        break
    
    
    raise #below is just for a one and done not stratifying
    print('stop')
    Tk().withdraw()
    
    processed_eeg_path=process_eeg()
    eeg_feats_filepath=make_eeg_feats(processed_eeg_path)
    train_set_path, test_set_path = split_train_test(eeg_feats_filepath)
    train(train_set_path)
    test(test_set_path=None)
    
    '''can do manually with the following steps'''
    #run script with the __main__ raising error immediately
    #run lines 94-ish (root dir etc)
    #train_eeg=process+eeg((working+'dev/EEG/')
    #eeg_featset=working+'EEG_001009011.csv'
    #feats.make_feats(train_eeg,eeg_featset,'eeg',period=1)
    #train_set_path, test_set_path = split_train_test(eeg_featset)
    #train(train_set_path)
    #test(test_set_path)
    y_true,y_distro,y_pred=test(test_set_path)
    #ConfusionMatrixDisplay.from_predictions(y_true,y_pred) #need to update to sklearn 1.0.x
    conf=confusion_matrix(y_true,y_pred)
    ConfusionMatrixDisplay(conf)
    plt.show()
    

