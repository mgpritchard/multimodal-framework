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
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, log_loss, confusion_matrix, ConfusionMatrixDisplay #plot_confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, Trials
from hyperopt.pyll import scope, stochastic
import pandas as pd
import time
import pickle
import statistics as stats



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
    '''use handleFeats make_feats on a dir of data for a manual featset gen.
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
    eeg_ppt['ID_stratID']=eeg_ppt['ID_run'].astype(str)+eeg_ppt['Label'].astype(str)+eeg_ppt['ID_gestrep'].astype(str)
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
    
    '''have verified with pd.merge on train and test that there is no
    duplication of rows (i.e. windows). in the matlab slicing, there may be
    one 0.002s datapoint that is found in both the start of grasp and the
    end of preceding rest etc. However that is 2 milliseconds of one 1 second
    window out of 5 - 10 seconds worth of grasp or rest'''

def single_mode_predict(test_set,model,classlabels,args):

    predlist=[]         
    '''Get values & targets from dataframe'''
    targets=test_set.pop('Label').values
    vals=ml.drop_ID_cols(test_set)
    
    '''Pass values to model'''
    distros=ml.prob_dist(model,vals)
    for distro in distros:
        predlist.append(ml.pred_from_distro(classlabels,distro))
        
    return targets, predlist

def classifyEEG_withinsubject(args):
    start=time.time()
    if args['data_in_memory']:
        eeg_set=args['eeg_set']
    else:
        eeg_set_path=args['eeg_set_path']
        eeg_set=ml.pd.read_csv(eeg_set_path,delimiter=',')
    if not args['prebalanced']:
        eeg_set=fus.balance_single_mode(eeg_set)
    eeg_masks=fus.get_ppt_split(eeg_set,args)
    
    accs=[]
    f1s=[]
    kappas=[]
    for idx,eeg_mask in enumerate(eeg_masks):
        eeg_ppt = eeg_set[eeg_mask]
        
        eeg_ppt['ID_stratID']=eeg_ppt['ID_run'].astype(str)+eeg_ppt['Label'].astype(str)+eeg_ppt['ID_gestrep'].astype(str)
        train_split,test_split=train_test_split(eeg_ppt['ID_stratID'].unique(),test_size=0.33)
        eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(train_split)]
        eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split)]
        
        if args['scalingtype']:
            eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
            eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
                
        eeg_train=ml.drop_ID_cols(eeg_train)
        sel_cols=feats.sel_percent_feats_df(eeg_train,percent=15)
        sel_cols=np.append(sel_cols,eeg_train.columns.get_loc('Label'))
        eeg_train=eeg_train.iloc[:,sel_cols]
        
        eeg_test=ml.drop_ID_cols(eeg_test) #this MUST BE DONE before iloc
        eeg_test=eeg_test.iloc[:,sel_cols]
        
        eeg_model = ml.train_optimise(eeg_train, args['eeg']['eeg_model_type'], args['eeg'])
        classlabels = eeg_model.classes_
        
        targets, predlist_eeg = single_mode_predict(eeg_test,eeg_model,classlabels,args)

        gest_truth=[params.idx_to_gestures[gest] for gest in targets]
        gest_pred_eeg=[params.idx_to_gestures[pred] for pred in predlist_eeg]
        
        if args['plot_confmats']:
            gesturelabels=[params.idx_to_gestures[label] for label in classlabels]
            tt.confmat(gest_truth,gest_pred_eeg,gesturelabels)
                   
        accs.append(accuracy_score(gest_truth,gest_pred_eeg))
        f1s.append(f1_score(gest_truth,gest_pred_eeg,average='weighted'))        
        kappas.append(cohen_kappa_score(gest_truth,gest_pred_eeg))
        
    mean_acc=stats.mean(accs)
    median_acc=stats.median(accs)
    mean_f1_fusion=stats.mean(f1s)
    median_f1=stats.median(f1s)
    median_kappa=stats.median(kappas)
    end=time.time()
    #return 1-mean_acc
    return {
        #'loss': 1-median_kappa,
        'loss':1-median_acc,
        'status': STATUS_OK,
        'median_kappa':median_kappa,
        'mean_acc':mean_acc,
        'median_acc':median_acc,
        'max_acc':max(accs),
        'max_acc_index':np.argmax(accs),
        'f1_mean':mean_f1_fusion,
        'elapsed_time':end-start,}


def classifyEEG_LOO(args):
    start=time.time()
    if args['data_in_memory']:
        eeg_set=args['eeg_set']
    else:
        eeg_set_path=args['eeg_set_path']
        eeg_set=ml.pd.read_csv(eeg_set_path,delimiter=',')
    if not args['prebalanced']:
        eeg_set=fus.balance_single_mode(eeg_set)
        
    eeg_masks=fus.get_ppt_split(eeg_set,args)
    
    accs=[]
    f1s=[]
    kappas=[]
    for idx,eeg_mask in enumerate(eeg_masks):
        eeg_ppt = eeg_set[eeg_mask]
        eeg_others = eeg_set[~eeg_mask]
        
        if args['scalingtype']:
            eeg_others,eegscaler=feats.scale_feats_train(eeg_others,args['scalingtype'])
            eeg_ppt=feats.scale_feats_test(eeg_ppt,eegscaler)
        
        eeg_others=ml.drop_ID_cols(eeg_others)
        
        sel_cols=feats.sel_percent_feats_df(eeg_others,percent=15)
        sel_cols=np.append(sel_cols,eeg_others.columns.get_loc('Label'))
        eeg_others=eeg_others.iloc[:,sel_cols]
        eeg_ppt=ml.drop_ID_cols(eeg_ppt) #this MUST BE DONE before iloc
        eeg_ppt=eeg_ppt.iloc[:,sel_cols]
        
        eeg_model = ml.train_optimise(eeg_others, args['eeg']['eeg_model_type'], args['eeg'])
        classlabels = eeg_model.classes_
        
        #eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        targets, predlist_eeg = single_mode_predict(eeg_ppt,eeg_model,classlabels,args)

        gest_truth=[params.idx_to_gestures[gest] for gest in targets]
        gest_pred_eeg=[params.idx_to_gestures[pred] for pred in predlist_eeg]

        if args['plot_confmats']:
            gesturelabels=[params.idx_to_gestures[label] for label in classlabels]
            tt.confmat(gest_truth,gest_pred_eeg,gesturelabels)
                   
        accs.append(accuracy_score(gest_truth,gest_pred_eeg))
        f1s.append(f1_score(gest_truth,gest_pred_eeg,average='weighted'))        
        kappas.append(cohen_kappa_score(gest_truth,gest_pred_eeg))
        
    mean_acc=stats.mean(accs)
    median_acc=stats.median(accs)
    mean_f1_fusion=stats.mean(f1s)
    median_f1=stats.median(f1s)
    median_kappa=stats.median(kappas)
    end=time.time()
    #return 1-mean_acc
    return {
        #'loss': 1-median_kappa,
        'loss':1-median_acc,
        'status': STATUS_OK,
        'median_kappa':median_kappa,
        'mean_acc':mean_acc,
        'median_acc':median_acc,
        'max_acc':max(accs),
        'max_acc_index':np.argmax(accs),
        'f1_mean':mean_f1_fusion,
        'elapsed_time':end-start,}

    
def setup_search_space():
    space = {
            'eeg':hp.choice('eeg model',[
                {'eeg_model_type':'RF',
                 'n_trees':scope.int(hp.quniform('eeg_ntrees',10,100,q=5)),
                 },
 #               {'eeg_model_type':'kNN',
  #               'knn_k':scope.int(hp.quniform('eeg.knn.k',1,25,q=1)),
   #              },
                {'eeg_model_type':'LDA',
                 'LDA_solver':hp.choice('eeg.LDA_solver',['svd','lsqr','eigen']),
                 'shrinkage':hp.uniform('eeg.lda.shrinkage',0.0,1.0),
                 },
                {'eeg_model_type':'QDA',
                 'regularisation':hp.uniform('eeg.qda.regularisation',0.0,1.0),
                 },
             #   {'eeg_model_type':'SVM',
              #   'svm_C':hp.uniform('eeg.svm.c',0.1,100),
                 # naming convention https://github.com/hyperopt/hyperopt/issues/380#issuecomment-685173200
              #   }
                ]),
            'eeg_set_path':params.eeg_jeongCSP_feats,
            'using_literature_data':True,
            'data_in_memory':False,
            'prebalanced':False,
            #'scalingtype':hp.choice('scaling',['normalise','standardise',None]),
            'scalingtype':None,
            'plot_confmats':False,
            }
    return space

def optimise_EEG_LOO(prebalance=True):
    space=setup_search_space()
    
    if prebalance:
        eeg_set=ml.pd.read_csv(space['eeg_set_path'],delimiter=',')
        eeg_set=fus.balance_single_mode(eeg_set)
        space.update({'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True})
    
    trials=Trials() #http://hyperopt.github.io/hyperopt/getting-started/minimizing_functions/#attaching-extra-information-via-the-trials-object
    best = fmin(classifyEEG_LOO,
                space=space,
                algo=tpe.suggest,
                max_evals=20,
                trials=trials)
    return best, space, trials

def optimise_EEG_withinsubject(prebalance=True):
    space=setup_search_space()
    
    if prebalance:
        eeg_set=ml.pd.read_csv(space['eeg_set_path'],delimiter=',')
        eeg_set=fus.balance_single_mode(eeg_set)
        space.update({'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True})
    
    trials=Trials() #http://hyperopt.github.io/hyperopt/getting-started/minimizing_functions/#attaching-extra-information-via-the-trials-object
    best = fmin(classifyEEG_withinsubject,
                space=space,
                algo=tpe.suggest,
                max_evals=20,
                trials=trials)
    return best, space, trials
    
def save_resultdict(filepath,resultdict):
    #https://stackoverflow.com/questions/61894745/write-dictionary-to-text-file-with-newline
    status=resultdict['Results'].pop('status')
    f=open(filepath,'w')
    try:
        target=list(resultdict['Results'].keys())[list(resultdict['Results'].values()).index(1-resultdict['Results']['loss'])]
        f.write(f"Optimising for {target}\n\n")
    except ValueError:
        target, _ = min(resultdict['Results'].items(), key=lambda x: abs(1-resultdict['Results']['loss'] - x[1]))
        f.write(f"Probably optimising for {target}\n\n")
    f.write('EEG Parameters:\n')
    for k in resultdict['Chosen parameters']['eeg'].keys():
        f.write(f"\t'{k}':'{resultdict['Chosen parameters']['eeg'][k]}'\n")
    f.write('Results:\n')
    resultdict['Results']['status']=status
    for k in resultdict['Results'].keys():
        f.write(f"\t'{k}':'{resultdict['Results'][k]}'\n")
    f.close()
    

## Testing the suspiciously hihgh accuracy:
#load model with testset,testset_attribs=ml.matrixdropID(testsetfile.csv)
#attrib_names = list(testset_attribs)
#plt.figure()
#tree.plot_tree(model,feature_names=attrib_names,max_depth=2,fontsize=6)    

if __name__ == '__main__':
    
    trialmode='LOO'
    
    if trialmode=='LOO':
        best,space,trials=optimise_EEG_LOO()
    elif trialmode=='WithinPpt':
        best,space,trials=optimise_EEG_withinsubject()
    
    if 0:
        '''performing a whole fresh evaluation with the chosen params'''
        best_results=classifyEEG_LOO(space_eval(space,best))
    else:
        best_results=trials.best_trial['result']
        #https://stackoverflow.com/questions/20776502/where-to-find-the-loss-corresponding-to-the-best-configuration-with-hyperopt
    #could just get trials.results?
    
    if 1:    
        chosen_space=space_eval(space,best)
        chosen_space['plot_confmats']=True
        if trialmode=='LOO':
            chosen_results=classifyEEG_LOO(chosen_space)
        elif trialmode=='WithinPpt':
            chosen_results=classifyEEG_withinsubject(chosen_space)
    
    bestparams=space_eval(space,best)
    print(bestparams)
    print('Best Coehns Kappa between ground truth and predictions: ',
          best_results['median_kappa'])
    #https://datascience.stackexchange.com/questions/24372/low-kappa-score-but-high-accuracy
    
    for static in ['eeg_set_path','using_literature_data']:
        bestparams.pop(static)
        
    winner={'Chosen parameters':bestparams,
            'Results':best_results}
    
    eeg_acc_plot=fus.plot_stat_in_time(trials, 'mean_acc',showplot=False)
    #plot_stat_in_time(trials, 'loss')
    #fus.plot_stat_in_time(trials,'elapsed_time',0,200)
    '''
    table=pd.DataFrame(trials.trials)
    table_readable=pd.concat(
        [pd.DataFrame(table['result'].tolist()),
         pd.DataFrame(pd.DataFrame(table['misc'].tolist())['vals'].values.tolist())],
        axis=1,join='outer')
    '''
    #print('plotting ppt1 just to get a confmat')
    #ppt1acc=function_fuse_pptn(space_eval(space,best),1,plot_confmats=True)
    
    currentpath=os.path.dirname(__file__)
    result_dir=params.jeong_results_dir
    resultpath=os.path.join(currentpath,result_dir,'EEGCSP',trialmode)
        
    '''saving figures of performance over time'''
    eeg_acc_plot.savefig(os.path.join(resultpath,'eeg_acc.png'))
    
    '''saving best parameters & results'''
    reportpath=os.path.join(resultpath,'params_results_report.txt')
    save_resultdict(reportpath,winner)
        
    raise KeyboardInterrupt('ending execution here!')
    
    
    
    
    '''
    WAYGAL_P4_set='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/WAYGAL_P4_Feats.csv'
    WAYGAL_P4_8ch_set='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/WAYGAL_P4_8channelFeats.csv'
    
    eeg_wayg_set='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/P4_CSVs/P4_EEG8chFeats.csv'
    emg_wayg_set='/home/michael/Documents/Aston/EEG/WAY-EEG-GAL Data/P4_CSVs/P4_EMGFeats.csv'
    
    eeg_within_ppt(eeg_wayg_set,single_ppt_dataset=True)
    print('below is EMG')
    eeg_within_ppt(emg_wayg_set,single_ppt_dataset=True)
    raise
    '''
    
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
    

