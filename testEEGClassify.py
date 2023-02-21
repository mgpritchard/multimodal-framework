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
import handleTrainTestPipeline as tt
import testFusion as fus
import params
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay #plot_confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



'''def process_eeg(datain=None): #deprecated, now moved to handleTrainTestPipeline as process_data
    if datain is None:
        datain=askdirectory(initialdir=root,title='Select EEG Directory')
    dataout=datain
    datacolsrearr=datain
    wrangle.process_eeg(datain,datacolsrearr,dataout)
    #sync_raw_files()    #not yet ready as EEG data not ready
    print('**processed raw eeg**')
    return dataout'''

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
    train,test=train_test_split(featset,test_size=0.25,shuffle=False)
    print('**split train/test**')

    train_path=featspath[:-4]+'_trainslice.csv'
    test_path=featspath[:-4]+'_testslice.csv'
    np.savetxt(train_path, train, delimiter = ',')
    np.savetxt(test_path, test, delimiter = ',')
    print('**saved train/test splits')
    return train_path, test_path


def eeg_within_ppt(eeg_set_path=None,single_ppt_dataset=False,selected_ppt=1):
    '''use handleFeats make_feats on a dir of data for a manual featset gen
    selected_ppt not needed if single_ppt_dataset is True'''
    
    args={'eeg_model_type':'RF',
                 'n_trees':20}
    if eeg_set_path is None:
        eeg_set_path='/home/michael/Documents/Aston/MultimodalFW/working_dataset/devset_EEG/featsEEGNewDecImpulseKill.csv'
    eeg_set=ml.pd.read_csv(eeg_set_path,delimiter=',')
    
    if not single_ppt_dataset:
        eeg_masks=fus.get_ppt_split(eeg_set)
        ppt=eeg_masks[selected_ppt]
        eeg_ppt = eeg_set[ppt]
    else:
        eeg_ppt=eeg_set
    
    #eeg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    #index_eeg=ml.pd.MultiIndex.from_arrays([eeg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    eeg_ppt['ID_stratID']=eeg_ppt['ID_run']+eeg_ppt['Label']+eeg_ppt['ID_gestrep']
    train_split,test_split=train_test_split(eeg_ppt['ID_stratID'].unique(),test_size=0.33)
    eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(train_split)]
    eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split)]
    eeg_train=ml.drop_ID_cols(eeg_train)
    
    eeg_model_type=args['eeg_model_type']
    eeg_model = ml.train_optimise(eeg_train, eeg_model_type, args)
    classlabels = eeg_model.classes_
    
    eeg_test_set=ml.drop_ID_cols(eeg_test)
    eeg_test_labels=eeg_test_set['Label'].values
    eeg_test_set=eeg_test_set.drop(['Label'],axis=1)
    preds=eeg_model.predict(eeg_test_set)
    acc=accuracy_score(eeg_test_labels,preds)
    print(acc)
    tt.confmat(eeg_test_labels,preds,classlabels)
    

'''def train(train_path): #deprecated, now moved to handleTrainTestPipeline
    ml.train_offline('RF',train_path)
    print('**trained a model**')'''

'''def test(test_set_path=None): #deprecated, now moved to handleTrainTestPipeline
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
    
    confmat(gest_truth,gest_pred,gesturelabels,testset=test_set_path)
    return gest_truth,distrolist,gest_pred'''

'''def confmat(y_true,y_pred,labels,modelname="",testset=""): #deprecated, now moved to handleTrainTestPipeline
    conf=confusion_matrix(y_true,y_pred,labels=labels)
    cm=ConfusionMatrixDisplay(conf,labels).plot()
    cm.figure_.suptitle=(modelname+'\n'+testset)
    plt.show()'''
    
'''def copy_files(filelist,emg_dest,eeg_dest): #deprecated, now moved to handleTrainTestPipeline
    for file in filelist:
        if file[-7:-4]=='EEG':
            dest=os.path.join(eeg_dest,file)
        else:
            dest=os.path.join(emg_dest,file)
        source=os.path.join(path,file)
        if not os.path.exists(dest):
            comp.copyfile(source,dest)'''

'''def ditch_EEG_suffix(eegdir): #deprecated, now moved to handleComposeDataset
    for file in os.listdir(eegdir):
        if file.endswith('_EEG',0,-4):
            os.remove(os.path.join(eegdir,file))'''


## Testing the suspiciously hihgh accuracy:
#load model with testset,testset_attribs=ml.matrixdropID(testsetfile.csv)
#attrib_names = list(testset_attribs)
#plt.figure()
#tree.plot_tree(model,feature_names=attrib_names,max_depth=2,fontsize=6)    

if __name__ == '__main__':
    
    WAYGAL_P4_set='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/WAYGAL_P4_Feats.csv'
    eeg_within_ppt(WAYGAL_P4_set,single_ppt_dataset=True)
    raise
    
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
        
        tt.copy_files(trainfiles,train_emg,train_eeg)
        tt.copy_files(testfiles,test_emg,test_eeg)
        
        '''ONLY DO THIS IF NOT ALREADY PROCESSED EEG'''
        if 1:   
            train_eeg = tt.process_data('eeg',train_eeg)
            comp.ditch_EEG_suffix(train_eeg)
            test_eeg = tt.process_data('eeg',test_eeg)
            comp.ditch_EEG_suffix(test_eeg)
        
        train_eeg_featset=working+str(pptnum)+'_eeg_train.csv'
        test_eeg_featset=working+str(pptnum)+'_eeg_test.csv'
        
        eeg_train_feats=feats.make_feats(train_eeg,train_eeg_featset,'eeg',period=1)
        eeg_train_feats=feats.select_feats(eeg_train_feats)
        eeg_test_feats=feats.make_feats(test_eeg,test_eeg_featset,'eeg',period=1)
        eeg_test_feats=feats.select_feats(eeg_test_feats)
        
        #eegtrain_labelled=train_eeg_featset[:-4] + '_Labelled.csv'
        #eegtest_labelled=test_eeg_featset[:-4] + '_Labelled.csv'
        tt.train(train_eeg_featset)
        y_true,y_distro,y_pred=tt.test('eeg',test_eeg_featset)
        #conf=confusion_matrix(y_true,y_pred)
        break #cutting off after 1 ppt for speedier testing
    
    
    raise #below is just for a one and done not stratifying
    print('stop')
    Tk().withdraw()
    
    processed_eeg_path=process_eeg()
    eeg_feats_filepath=make_eeg_feats(processed_eeg_path)
    train_set_path, test_set_path = split_train_test(eeg_feats_filepath)
    tt.train(train_set_path)
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
    

