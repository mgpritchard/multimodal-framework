# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 17:16:23 2023

@author: pritcham
"""


import sklearn as skl
from sklearn.base import BaseEstimator
import testfusion as fuse
import params as params
from sklearn.model_selection import train_test_split
import random

class optimal_bespoke_decision(BaseEstimator):
    
    def __init__(self,
                 params={'fusion_alg':'LDA',
                       'ldafuse':{'LDA_solver':'eigen','shrinkage':0.21678705982755336,},
                       'eeg':{'eeg_model_type':'kNN','knn_k':18,},
                       'emg':{'emg_model_type':'SVM_PlattScale',
                              'kernel':'rbf',
                              'svm_C':5.643617263738588,
                              'gamma':0.014586498446354922,},
                       'stack_distros':True,
                       'trialmode':'WithinPpt',
                       'l1_sparsity':0.005,
                       'l1_maxfeats':40},
                 architecture='decision'):


        space=fuse.setup_search_space(architecture,include_svm=True)
        space.update(params)
        emg_set=fuse.ml.pd.read_csv(params.jeong_EMGfeats,delimiter=',')
        eeg_set=fuse.ml.pd.read_csv(params.jeong_noCSP_WidebandFeats,delimiter=',')
        emg_set,eeg_set=fuse.balance_set(emg_set,eeg_set)
        space.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,})
        self._estimator_type = "classifier"
        self.params=params
        self.args = space
        
    def fit(self, X, y, sample_weight=None):
        emg_train=self.args['emg_set'][X]
        eeg_train=self.args['eeg_set'][X]
        
        emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        
        index_emg=fuse.ml.pd.MultiIndex.from_arrays([emg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
        index_eeg=fuse.ml.pd.MultiIndex.from_arrays([eeg_train[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
        emg_train=emg_train.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
        eeg_train=eeg_train.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
                
        emg_train['ID_splitIndex']=emg_train['Label'].astype(str)+emg_train['ID_pptID'].astype(str)
        eeg_train['ID_splitIndex']=eeg_train['Label'].astype(str)+eeg_train['ID_pptID'].astype(str)
        #https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
        random_split=random.randint(0,100)
        emg_train_split_ML,emg_train_split_fusion=train_test_split(emg_train,test_size=0.33,random_state=random_split,stratify=emg_train[['ID_splitIndex']])
        eeg_train_split_ML,eeg_train_split_fusion=train_test_split(eeg_train,test_size=0.33,random_state=random_split,stratify=eeg_train[['ID_splitIndex']])
        #https://stackoverflow.com/questions/43095076/scikit-learn-train-test-split-can-i-ensure-same-splits-on-different-datasets
        
        
        if self.args['scalingtype']:
                emg_train_split_ML,emgscaler=fuse.feats.scale_feats_train(emg_train_split_ML,self.args['scalingtype'])
                eeg_train_split_ML,eegscaler=fuse.feats.scale_feats_train(eeg_train_split_ML,self.args['scalingtype'])
                emg_train_split_fusion=fuse.feats.scale_feats_test(emg_train_split_fusion,emgscaler)
                eeg_train_split_fusion=fuse.feats.scale_feats_test(eeg_train_split_fusion,eegscaler)
                emg_test=fuse.feats.scale_feats_test(emg_test,emgscaler)
                eeg_test=fuse.feats.scale_feats_test(eeg_test,eegscaler)
    
        emg_train_split_ML=ml.drop_ID_cols(emg_train_split_ML)
        eeg_train_split_ML=ml.drop_ID_cols(eeg_train_split_ML)
            
        sel_cols_emg=fuse.feats.sel_percent_feats_df(emg_train_split_ML,percent=15)
        sel_cols_emg=np.append(sel_cols_emg,emg_train_split_ML.columns.get_loc('Label'))
        emg_train_split_ML=emg_train_split_ML.iloc[:,sel_cols_emg]
        
        #sel_cols_eeg=feats.sel_percent_feats_df(eeg_train_split_ML,percent=3)
        sel_cols_eeg=fuse.feats.sel_feats_l1_df(eeg_train_split_ML,sparsityC=self.args['l1_sparsity'],maxfeats=self.args['l1_maxfeats'])
        sel_cols_eeg=np.append(sel_cols_eeg,eeg_train_split_ML.columns.get_loc('Label'))
        eeg_train_split_ML=eeg_train_split_ML.iloc[:,sel_cols_eeg]
           
        emg_model,eeg_model=train_models_opt(emg_train_split_ML,eeg_train_split_ML,self.args)
        
        classlabels = emg_model.classes_
            
        
        
def fusion_LDA(emg_train, eeg_train, emg_test, eeg_test, args):
    

    emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
    targets, predlist_emg, predlist_eeg, _, distros_emg, distros_eeg, _ = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
    
    fuser,onehotEncoder=train_lda_fuser(emg_model,eeg_model,emg_train_split_fusion,eeg_train_split_fusion,classlabels,args,sel_cols_eeg,sel_cols_emg)
    if args['stack_distros']:
        predlist_fusion=fusion.lda_fusion(fuser,onehotEncoder,distros_emg,distros_eeg,classlabels)
    else:
        predlist_fusion=fusion.lda_fusion(fuser,onehotEncoder,predlist_emg,predlist_eeg,classlabels)
    
    if args['get_train_acc']:
        emg_train=feats.scale_feats_test(emg_train,emgscaler)
        eeg_train=feats.scale_feats_test(eeg_train,eegscaler)
        emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
        traintargs, predlist_emgtrain, predlist_eegtrain, _, distros_emgtrain, distros_eegtrain, _ = refactor_synced_predict(emg_train, eeg_train, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
        if args['stack_distros']:
            predlist_train=fusion.lda_fusion(fuser,onehotEncoder,distros_emgtrain,distros_eegtrain,classlabels)
        else:
            predlist_train=fusion.lda_fusion(fuser,onehotEncoder,predlist_emgtrain,predlist_eegtrain,classlabels)
        return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train  
    else:
        return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels
