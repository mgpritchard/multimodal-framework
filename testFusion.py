#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 23:42:00 2022

@author: pritcham
"""

import os
import numpy as np
import handleDatawrangle as wrangle
import handleFeats as feats
import handleML as ml
import handleComposeDataset as comp
import handleTrainTestPipeline as tt
import handleFusion as fusion
import params
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #plot_confusion_matrix
import matplotlib.pyplot as plt



if __name__ == '__main__':
    #raise
    '''Build or find a dataset, separating train and test'''

    


    '''Process the data *including* cropping to time'''
    #however might want to be able to look at EEG from t-1.
    
    emg_list=wrangle.list_raw_files('/home/michael/Documents/Aston/MultimodalFW/working_dataset/004 (copy)/trainsplit/EMG')
    eeg_list=wrangle.list_raw_files('/home/michael/Documents/Aston/MultimodalFW/working_dataset/004 (copy)/trainsplit/EEG')
    crop_eeg_dir='/home/michael/Documents/Aston/MultimodalFW/working_dataset/004 (copy)/trainsplit/Cropped EEG'
    crop_emg_dir='/home/michael/Documents/Aston/MultimodalFW/working_dataset/004 (copy)/trainsplit/Cropped EMG'
    
    emg_TEST_list=wrangle.list_raw_files('/home/michael/Documents/Aston/MultimodalFW/working_dataset/004 (copy)/testsplit/EMG')
    eeg_TEST_list=wrangle.list_raw_files('/home/michael/Documents/Aston/MultimodalFW/working_dataset/004 (copy)/testsplit/EEG')
    crop_eeg_TEST_dir='/home/michael/Documents/Aston/MultimodalFW/working_dataset/004 (copy)/testsplit/Cropped EEG'
    crop_emg_TEST_dir='/home/michael/Documents/Aston/MultimodalFW/working_dataset/004 (copy)/testsplit/Cropped EMG'
    
    #needed a custom version of syncstarts etc that multiplies EEG start time by 1000
    #as EMG is in unix millis with ms precision but EEG is in unix secs (with ns precision)
    wrangle.sync_raw_files(emg_list,eeg_list,crop_emg_dir,crop_eeg_dir,unicorn_time_moved=1)
    wrangle.sync_raw_files(emg_TEST_list,eeg_TEST_list,crop_emg_TEST_dir,crop_eeg_TEST_dir,unicorn_time_moved=1)
    
    proc_emg_dir=tt.process_data('emg',crop_emg_dir,overwrite=False)
    proc_eeg_dir=tt.process_data('eeg',crop_eeg_dir,overwrite=False,bf_time_moved=True)
    
    proc_emg_TEST_dir=tt.process_data('emg',crop_emg_TEST_dir,overwrite=False)
    proc_eeg_TEST_dir=tt.process_data('eeg',crop_eeg_TEST_dir,overwrite=False,bf_time_moved=True)
    
    
    '''Generate features'''
    
    train_eeg_featset='/home/michael/Documents/Aston/MultimodalFW/working_dataset/004 (copy)/FusionEEGTrain.csv'
    train_emg_featset='/home/michael/Documents/Aston/MultimodalFW/working_dataset/004 (copy)/FusionEMGTrain.csv'
    
    eeg_train_feats=feats.make_feats(proc_eeg_dir,train_eeg_featset,'eeg',period=1000)
    emg_train_feats=feats.make_feats(proc_emg_dir,train_emg_featset,'emg',period=1000)
    
    test_eeg_featset='/home/michael/Documents/Aston/MultimodalFW/working_dataset/004 (copy)/FusionEEGTest.csv'
    test_emg_featset='/home/michael/Documents/Aston/MultimodalFW/working_dataset/004 (copy)/FusionEMGTest.csv'
    
    eeg_test_feats=feats.make_feats(proc_eeg_TEST_dir,test_eeg_featset,'eeg',period=1000)
    emg_test_feats=feats.make_feats(proc_emg_TEST_dir,test_emg_featset,'emg',period=1000)
    
    
    '''SELECT FEATURES'''
    #feats.select_feats(featureset)
    eeg_train_feats=feats.select_feats(eeg_train_feats)
    emg_train_feats=feats.select_feats(emg_train_feats)
    
    
    '''Train a model or models'''
    #classlabels=model.classes_
    tt.train(train_eeg_featset)
    tt.train(train_emg_featset)
    
    
    '''Load the testing dataset WITH ID columns'''
    
    ''' if testing JUST FROM HERE then FIRST need to run the following lines'''
    #test_eeg_featset='/home/michael/Documents/Aston/MultimodalFW/working_dataset/004 (copy)/FusionEEGTest.csv'
    #test_emg_featset='/home/michael/Documents/Aston/MultimodalFW/working_dataset/004 (copy)/FusionEMGTest.csv'
    '''as well as all the imports etc'''
    '''can then highlight the rest from here below & rclick -> run selection'''
    '''would be helpful to test if any of lines in testset match those in trainset'''
    '''just in case that eeg accuracy is too high'''
    
    
    test_set_emg=ml.pd.read_csv(test_emg_featset,delimiter=",")
    test_set_eeg=ml.pd.read_csv(test_eeg_featset,delimiter=",")
    
    
    '''Sort test set by ppt, then by run, then by gesture, then by rep'''    
    test_set_emg.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    test_set_eeg.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    #pandas: dataframe.sort_values(['column','names']), ascending=[True,False],inplace=True)
    
    
    '''Load a model or models'''
    root='/home/michael/Documents/Aston/MultimodalFW/'
    model_emg = ml.load_model('testing emg',root)
    model_eeg = ml.load_model('testing eeg',root)
    classlabels=model_emg.classes_
    
    '''*Step through feature instances and classify*'''
    
    distrolist_emg=[]
    predlist_emg=[]
    correctness_emg=[]
    
    distrolist_eeg=[]
    predlist_eeg=[]
    correctness_eeg=[]
    
    distrolist_fusion=[]
    predlist_fusion=[]
    correctness_fusion=[]
    
    targets=[]
    
    for index,emgrow in test_set_emg.iterrows():
        eegrow = test_set_eeg[(test_set_eeg['ID_pptID']==emgrow['ID_pptID'])
                              & (test_set_eeg['ID_run']==emgrow['ID_run'])
                              & (test_set_eeg['Label']==emgrow['Label'])
                              & (test_set_eeg['ID_gestrep']==emgrow['ID_gestrep'])
                              & (test_set_eeg['ID_tend']==emgrow['ID_tend'])]
        #syntax like the below would do it closer to a .where
        #eegrow=test_set_eeg[test_set_eeg[['ID_pptID','Label']]==emgrow[['ID_pptID','Label']]]
        if eegrow.empty:
            print('No matching EEG window for EMG window '+emgrow['ID_pptID']+emgrow['ID_run']+emgrow['Label']+emgrow['ID_gestrep']+emgrow['ID_tend'])
            continue
        
        TargetLabel=emgrow['Label']
        if TargetLabel != eegrow['Label'].values:
            raise Exception('Sense check failed, target label should agree between modes')
        
        '''Get values from instances'''
        IDs=list(emgrow.filter(regex='^ID_').keys())
        IDs.append('Label')
        emgvals=emgrow.drop(IDs).values
        eegvals=eegrow.drop(IDs,axis='columns').values
        
        '''Pass values to models'''
        
        distro_emg=ml.prob_dist(model_emg,emgvals.reshape(1,-1))
        pred_emg=ml.pred_from_distro(classlabels,distro_emg)
        distrolist_emg.append(distro_emg)
        predlist_emg.append(pred_emg)
        
        if pred_emg == TargetLabel:
            correctness_emg.append(True)
        else:
            correctness_emg.append(False)
        
        distro_eeg=ml.prob_dist(model_eeg,eegvals.reshape(1,-1))
        pred_eeg=ml.pred_from_distro(classlabels,distro_eeg)
        distrolist_eeg.append(distro_eeg)
        predlist_eeg.append(pred_eeg)
        
        if pred_eeg == TargetLabel:
            correctness_eeg.append(True)
        else:
            correctness_eeg.append(False)
        
        distro_fusion=fusion.fuse_mean(distro_emg,distro_eeg)
        pred_fusion=ml.pred_from_distro(classlabels,distro_fusion)
        distrolist_fusion.append(distro_fusion)
        predlist_fusion.append(pred_fusion)
        
        if pred_fusion == TargetLabel:
            correctness_fusion.append(True)
        else:
            correctness_fusion.append(False)
            
        targets.append(TargetLabel)
    
    #testset_values = test[:,0:-1]
    #testset_labels = test[:,-1]
    
    #distrolist=[]
    #predlist=[]
    #correctness=[]
    #for count,_ in enumerate(testset):
        #instance_emg=testset[count]
        #instance_eeg=testset[count]
        #distro_emg=ml.prob_dist(model_emg, instance.reshape(1,-1)) 
        #get the single mode too and save it for later comparisons
        #distro_eeg=ml.prob_dist(model_eeg, instance.reshape(1,-1))
        #get the single mode too and save it for comparisons
        #distro_fusion=fusion.fuse(distro_emg,distro_eeg)
        #prediction=ml.pred_from_distro(classlabels,distro_fusion)
        #distrolist.append(distro_fusion)
        #predlist.append(prediction)
        #if predlabel == testset_labels[inst_count]:
        #    correctness.append(True)
        #else:
        #    correctness.append(False)
    
    '''Evaluate accuracy'''
    accuracy_emg = sum(correctness_emg)/len(correctness_emg)
    print('EMG accuracy: '+ str(accuracy_emg))
    
    accuracy_eeg = sum(correctness_eeg)/len(correctness_eeg)
    print('EEG accuracy: '+str(accuracy_eeg))
    
    accuracy_fusion = sum(correctness_fusion)/len(correctness_fusion)
    print('Fusion accuracy: '+str(accuracy_fusion))
    
    '''Convert predictions to gesture labels'''
    gest_truth=[params.idx_to_gestures[gest] for gest in targets]
    gest_pred_emg=[params.idx_to_gestures[pred] for pred in predlist_emg]
    gest_pred_eeg=[params.idx_to_gestures[pred] for pred in predlist_eeg]
    gest_pred_fusion=[params.idx_to_gestures[pred] for pred in predlist_fusion]
    gesturelabels=[params.idx_to_gestures[label] for label in classlabels]
    
    '''Produce confusion matrix'''
    tt.confmat(gest_truth,gest_pred_emg,gesturelabels)
    tt.confmat(gest_truth,gest_pred_eeg,gesturelabels)
    tt.confmat(gest_truth,gest_pred_fusion,gesturelabels)#,testset=test_set_path)
    #CAN you have a consistent gradation of the colour heatmap across confmats?
    #ie yellow is always a fixed % not relative to the highest in that given
    #confmat
    
    '''End of protocol'''
    
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
        
        feats.make_feats(train_eeg,train_eeg_featset,'eeg',period=1)
        feats.make_feats(test_eeg,test_eeg_featset,'eeg',period=1)
        
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
    

