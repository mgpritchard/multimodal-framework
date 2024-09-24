# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 18:06:17 2023

@author: pritcham
"""
import time
import testFusion as fuse
import handleML as ml
import handleFeats as feats
import numpy as np
import statistics as stats
import random
from sklearn.model_selection import StratifiedKFold



def fuse_bespoke_kfold(args,train_wholeset_size,nfolds=5):
    
    if not args['data_in_memory']:
        emg_set_path=args['emg_set_path']
        eeg_set_path=args['eeg_set_path']
    
        emg_set=ml.pd.read_csv(emg_set_path,delimiter=',')
        eeg_set=ml.pd.read_csv(eeg_set_path,delimiter=',')
    else:
        emg_set=args['emg_set']
        eeg_set=args['eeg_set']
    if not args['prebalanced']: 
        emg_set,eeg_set=fuse.balance_set(emg_set,eeg_set)
    
    eeg_masks=fuse.get_ppt_split(eeg_set,args)
    emg_masks=fuse.get_ppt_split(emg_set,args)
    
    mean_ppt_acc=[]
    mean_ppt_emg=[]
    mean_ppt_eeg=[]
    mean_ppt_kappa=[]
    mean_ppt_train_acc=[]
    mean_ppt_duration=[]

    for idx,emg_mask in enumerate(emg_masks):

        eeg_mask=eeg_masks[idx]
        
        emg_ppt = emg_set[emg_mask]
        eeg_ppt = eeg_set[eeg_mask]
        
        emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        
        index_emg=ml.pd.MultiIndex.from_arrays([emg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
        index_eeg=ml.pd.MultiIndex.from_arrays([eeg_ppt[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
        emg_ppt=emg_ppt.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
        eeg_ppt=eeg_ppt.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
        
        eeg_ppt['ID_stratID']=eeg_ppt['ID_run'].astype(str)+eeg_ppt['Label'].astype(str)+eeg_ppt['ID_gestrep'].astype(str)
        emg_ppt['ID_stratID']=emg_ppt['ID_run'].astype(str)+eeg_ppt['Label'].astype(str)+eeg_ppt['ID_gestrep'].astype(str)
        random_split=random.randint(0,100)
        if not emg_ppt['ID_stratID'].equals(eeg_ppt['ID_stratID']):
            raise ValueError('EMG & EEG performances misaligned')
        gest_perfs=emg_ppt['ID_stratID'].unique()
        gest_strat=fuse.pd.DataFrame([gest_perfs,[perf.split('.')[1][-1] for perf in gest_perfs]]).transpose()
        
        folds=StratifiedKFold(random_state=random_split,n_splits=nfolds,shuffle=True)
        ppt_duration=[]
        accs=[]
        emg_accs=[]
        eeg_accs=[]  
        train_accs=[]
        kappas=[]
        for i, (train_index, test_index) in enumerate(folds.split(gest_strat,gest_strat[1])):


            start=time.time()
            #train_split,test_split=fuse.train_test_split(gest_strat,test_size=1-train_size,random_state=random_split,stratify=gest_strat[1])
            train_split=gest_strat.iloc[train_index]
            train_split_scaled,_=fuse.train_test_split(train_split,train_size=train_wholeset_size/(1-1/nfolds),random_state=random_split,stratify=train_split[1])
            test_split=gest_strat.iloc[test_index]
            
            eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(train_split_scaled[0])]
            eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split[0])]
            emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(train_split_scaled[0])]
            emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]
            '''
            eeg_train=eeg_ppt.iloc[train_index]
            emg_train=emg_ppt.iloc[train_index]
            eeg_test=eeg_ppt.iloc[test_index]
            emg_test=emg_ppt.iloc[test_index]
            '''
    
            
            if args['fusion_alg']=='svm':
                if args['get_train_acc']:
                    targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train = fuse.fusion_SVM(emg_train, eeg_train, emg_test, eeg_test, args)
                else:
                    targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.fusion_SVM(emg_train, eeg_train, emg_test, eeg_test, args)
            
            elif args['fusion_alg']=='lda':
                if args['get_train_acc']:
                    targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train = fuse.fusion_LDA(emg_train, eeg_train, emg_test, eeg_test, args)
                else:
                    targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.fusion_LDA(emg_train, eeg_train, emg_test, eeg_test, args)
            
            elif args['fusion_alg']=='rf':
                if args['get_train_acc']:
                    targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train = fuse.fusion_RF(emg_train, eeg_train, emg_test, eeg_test, args)
                else:
                    targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.fusion_RF(emg_train, eeg_train, emg_test, eeg_test, args)
            
            elif args['fusion_alg']=='hierarchical':
                         
                if args['scalingtype']:
                    emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
                    eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
                    emg_test=feats.scale_feats_test(emg_test,emgscaler)
                    eeg_test=feats.scale_feats_test(eeg_test,eegscaler)                            
                            
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.fusion_hierarchical(emg_train, eeg_train, emg_test, eeg_test, args)
    
            elif args['fusion_alg']=='hierarchical_inv':
                
                if args['scalingtype']:
                    emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
                    eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
                    emg_test=feats.scale_feats_test(emg_test,emgscaler)
                    eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
    
                if not args['get_train_acc']:            
                    targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.fusion_hierarchical_inv(emg_train, eeg_train, emg_test, eeg_test, args)
                else:
                    targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train = fuse.fusion_hierarchical_inv(emg_train, eeg_train, emg_test, eeg_test, args)
                     
            elif args['fusion_alg']=='featlevel': 
                
                if args['scalingtype']:
                    emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
                    eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
                    emg_test=feats.scale_feats_test(emg_test,emgscaler)
                    eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
                                
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.feature_fusion(emg_train, eeg_train, emg_test, eeg_test, args)
            
            elif args['fusion_alg']=='just_emg':
                
                if args['scalingtype']:
                    emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
                    eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
                    emg_test=feats.scale_feats_test(emg_test,emgscaler)
                    eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
                
                if not args['get_train_acc']:
                    targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.only_EMG(emg_train, eeg_train, emg_test, eeg_test, args)
                else:
                    targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=fuse.only_EMG(emg_train, eeg_train, emg_test, eeg_test, args)
            
            elif args['fusion_alg']=='just_eeg':
                
                if args['scalingtype']:
                    emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
                    eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
                    emg_test=feats.scale_feats_test(emg_test,emgscaler)
                    eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
    
                if not args['get_train_acc']:    
                    targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.only_EEG(emg_train, eeg_train, emg_test, eeg_test, args)
                else:
                    targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=fuse.only_EEG(emg_train, eeg_train, emg_test, eeg_test, args)
            else:
                
                if args['scalingtype']:
                    emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
                    eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
                    emg_test=feats.scale_feats_test(emg_test,emgscaler)
                    eeg_test=feats.scale_feats_test(eeg_test,eegscaler)
                
                if args['get_train_acc']:
                    emg_trainacc=emg_train.copy()
                    eeg_trainacc=eeg_train.copy()
                    emg_trainacc.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
                    eeg_trainacc.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
               
                
                emg_train=ml.drop_ID_cols(emg_train)
                eeg_train=ml.drop_ID_cols(eeg_train)
                
                #sel_cols_eeg=feats.sel_percent_feats_df(eeg_train,percent=3)
                sel_cols_eeg=feats.sel_feats_l1_df(eeg_train,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
                sel_cols_eeg=np.append(sel_cols_eeg,eeg_train.columns.get_loc('Label'))
                eeg_train=eeg_train.iloc[:,sel_cols_eeg]
                
                sel_cols_emg=feats.sel_percent_feats_df(emg_train,percent=15)
                sel_cols_emg=np.append(sel_cols_emg,emg_train.columns.get_loc('Label'))
                emg_train=emg_train.iloc[:,sel_cols_emg]
                
                emg_model,eeg_model=fuse.train_models_opt(emg_train,eeg_train,args)
            
                classlabels = emg_model.classes_
                
                emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
                eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
                    
                targets, predlist_emg, predlist_eeg, predlist_fusion,_,_,_ = fuse.refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args, sel_cols_eeg,sel_cols_emg)
    
                if args['get_train_acc']:
                    traintargs, predlist_emgtrain, predlist_eegtrain, predlist_train,_,_,_ = fuse.refactor_synced_predict(emg_trainacc, eeg_trainacc, emg_model, eeg_model, classlabels, args, sel_cols_eeg,sel_cols_emg)
    
            #acc_emg,acc_eeg,acc_fusion=evaluate_results(targets, predlist_emg, correctness_emg, predlist_eeg, correctness_eeg, predlist_fusion, correctness_fusion, classlabels)
            
            gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels=fuse.classes_from_preds(targets,predlist_emg,predlist_eeg,predlist_fusion,classlabels)
            '''could calculate log loss if got the probabilities back''' #https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
                        
            if args['plot_confmats']:
                gesturelabels=[fuse.params.idx_to_gestures[label] for label in classlabels]
                fuse.tt.confmat(gest_truth,gest_pred_eeg,gesturelabels,title='EEG')
                fuse.tt.confmat(gest_truth,gest_pred_emg,gesturelabels,title='EMG')
                fuse.tt.confmat(gest_truth,gest_pred_fusion,gesturelabels,title='Fusion')
                
            emg_accs.append(fuse.accuracy_score(gest_truth,gest_pred_emg))
            eeg_accs.append(fuse.accuracy_score(gest_truth,gest_pred_eeg))
            accs.append(fuse.accuracy_score(gest_truth,gest_pred_fusion))
            
            kappas.append(fuse.cohen_kappa_score(gest_truth,gest_pred_fusion))
            
            if args['get_train_acc']:
                train_truth=[fuse.params.idx_to_gestures[gest] for gest in traintargs]
                train_preds=[fuse.params.idx_to_gestures[pred] for pred in predlist_train]
                train_accs.append(fuse.accuracy_score(train_truth,train_preds))
            else:
                train_accs.append(0)
                
            ppt_duration.append(time.time()-start)
                
        mean_ppt_acc.append(stats.mean(accs))
        mean_ppt_emg.append(stats.mean(emg_accs))
        mean_ppt_eeg.append(stats.mean(eeg_accs))
        mean_ppt_kappa.append(stats.mean(kappas))
        mean_ppt_train_acc.append(stats.mean(train_accs))
        mean_ppt_duration.append(stats.mean(ppt_duration))
        
    mean_acc=stats.mean(mean_ppt_acc)
    median_acc=stats.median(mean_ppt_acc)
    mean_emg=stats.mean(mean_ppt_emg)
    median_emg=stats.median(mean_ppt_emg)
    mean_eeg=stats.mean(mean_ppt_eeg)
    median_eeg=stats.median(mean_ppt_eeg)
    median_kappa=stats.median(mean_ppt_kappa)
    mean_train_acc=stats.mean(mean_ppt_train_acc)
    mean_durations=stats.mean(mean_ppt_duration)
    #return 1-mean_acc
    return {
        'loss': 1-mean_acc,
        'median_kappa':median_kappa,
        'fusion_mean_acc':mean_acc,
        'fusion_median_acc':median_acc,
        'emg_mean_acc':mean_emg,
        'emg_median_acc':median_emg,
        'eeg_mean_acc':mean_eeg,
        'eeg_median_acc':median_eeg,
        'emg_accs':mean_ppt_emg,
        'eeg_accs':mean_ppt_eeg,
        'fusion_accs':mean_ppt_acc,
        'mean_train_acc':mean_train_acc,
        'train_size':train_wholeset_size,
        'elapsed_time':mean_durations,}

if __name__ == '__main__':
    
    
    emg_set=fuse.ml.pd.read_csv(fuse.params.jeong_EMGfeats,delimiter=',')
    eeg_set=fuse.ml.pd.read_csv(fuse.params.jeong_noCSP_WidebandFeats,delimiter=',')
    emg_set,eeg_set=fuse.balance_set(emg_set,eeg_set)
    
    #train_sizes=np.linspace(0.1,0.67,10)
    '''try a log scale too'''
    train_sizes1=0.77-np.geomspace(0.1,0.2,5)[::-1]
    train_sizes_mid=np.linspace(0.2,0.57,5)
    train_sizes2=np.geomspace(0.01,0.2,10)
    train_sizes=np.unique(np.concatenate((train_sizes1,train_sizes_mid,train_sizes2)))
    #fuse.plt.plot(train_sizes,marker='o');fuse.plt.show()
    #fuse.plt.plot(train_sizes,np.zeros(len(train_sizes))+1,marker='o');fuse.plt.show()
    
    '''
    results_test=[]
    for train_wholeset_size in train_sizes:
        
        space_test=fuse.setup_search_space('decision',include_svm=True)
        space_test.update({'fusion_alg':'mean',
                               'eeg':{'eeg_model_type':'RF',
                                      'n_trees':2,
                                      'max_depth':5},
                               'emg':{'emg_model_type':'RF',
                                      'n_trees':2,
                                      'max_depth':5},
                               'stack_distros':True,
                               })
        space_test.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
    
        results_test.append(fuse_bespoke_kfold(space_test,train_wholeset_size))
    resultdf=ml.pd.DataFrame(results_test)
    resultdf.plot(x='train_size',y='fusion_mean_acc',style='o')
    '''
    results_emg=[]
    for train_wholeset_size in train_sizes:
        
        space_just_emg=fuse.setup_search_space('just_emg',include_svm=True)
        space_just_emg.update({'fusion_alg':'just_emg',
                      'emg':{'emg_model_type':'SVM_PlattScale',
                         'kernel':'rbf',
                         #'svm_C':10.8608, 
                         'svm_C':10.86077990984692,
                         #'gamma':0.0121,
                         'gamma':0.01207463833393443,
                         },
                      })
        space_just_emg.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
                               
        results_emg.append(fuse_bespoke_kfold(space_just_emg,train_wholeset_size))
    result_emg_df=ml.pd.DataFrame(results_emg)
    result_emg_df.plot(x='train_size',y='fusion_mean_acc',style='o',title='emg')
    result_emg_df.plot(x='train_size',y='fusion_mean_acc',style='o',title='emg',ylim=(0,1))
    
    
    
    results_eeg=[]
    for train_wholeset_size in train_sizes:
        
        space_just_eeg=fuse.setup_search_space('just_eeg',include_svm=True)
        space_just_eeg.update({'fusion_alg':'just_eeg',
                      'eeg':{'eeg_model_type':'LDA',
                         'LDA_solver':'lsqr',
                         #'shrinkage':0.2445, 
                         'shrinkage':0.24453863248230145,
                         },
                      })
        space_just_eeg.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
                               
        results_eeg.append(fuse_bespoke_kfold(space_just_eeg,train_wholeset_size))
    result_eeg_df=ml.pd.DataFrame(results_eeg)
    result_eeg_df.plot(x='train_size',y='fusion_mean_acc',style='o',title='eeg')
    result_eeg_df.plot(x='train_size',y='fusion_mean_acc',style='o',title='eeg',ylim=(0,1))
    
    
    
    results_decision=[]
    for train_wholeset_size in train_sizes:
        
        space_decision=fuse.setup_search_space('decision',include_svm=True)
        space_decision.update({'fusion_alg':'lda',
                           'ldafuse':{'LDA_solver':'eigen',
                                      #'shrinkage':0.216787
                                      'shrinkage':0.21678705982755336,},
                           'eeg':{'eeg_model_type':'kNN',
                                  'knn_k':18,},
                           'emg':{'emg_model_type':'SVM_PlattScale',
                                  'kernel':'rbf',
                                  #'svm_C':5.6436, 
                                  'svm_C':5.643617263738588,
                                  #'gamma':0.0146,
                                  'gamma':0.014586498446354922,},
                           'stack_distros':True,
                           })
        space_decision.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
                               
        results_decision.append(fuse_bespoke_kfold(space_decision,train_wholeset_size))
    result_decision_df=ml.pd.DataFrame(results_decision)
    result_decision_df.plot(x='train_size',y='fusion_mean_acc',style='o',title='decision')
    result_decision_df.plot(x='train_size',y='fusion_mean_acc',style='o',title='decision',ylim=(0,1))
    
    
    
    results_featlevel=[]
    for train_wholeset_size in train_sizes:
        
        space_featlevel=fuse.setup_search_space('featlevel',include_svm=True)
        space_featlevel.update({'fusion_alg':'featlevel',
                           'featfuse':
                               {'featfuse_model_type':'SVM_PlattScale', #keep this commented out
                                'kernel':'rbf',#'poly','linear']),
                                #'svm_C':3.6806,
                                'svm_C':3.680576929817659,
                                #'gamma':0.0107,
                                'gamma':0.010688027123522858,
                                },
                           'featfuse_sel_feats_together':False,
                           })
        space_featlevel.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
                               
        results_featlevel.append(fuse_bespoke_kfold(space_featlevel,train_wholeset_size))
    result_featlevel_df=ml.pd.DataFrame(results_featlevel)
    result_featlevel_df.plot(x='train_size',y='fusion_mean_acc',style='o',title='feat level sep select')
    result_featlevel_df.plot(x='train_size',y='fusion_mean_acc',style='o',title='feat level sep select',ylim=(0,1))
    
    
    
    results_featjoint=[]
    for train_wholeset_size in train_sizes:
        
        space_feat_jointsel=fuse.setup_search_space('featlevel',include_svm=True)
        space_feat_jointsel.update({'fusion_alg':'featlevel',
                           'featfuse':
                               {'featfuse_model_type':'SVM_PlattScale', #keep this commented out
                                'kernel':'rbf',#'poly','linear']),
                                #'svm_C':97.9778,
                                'svm_C':97.97784331407038,
                                #'gamma':0.0161,
                                'gamma':0.016112099194346027,
                                },
                           'featfuse_sel_feats_together':True,
                           })
        space_feat_jointsel.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
                               
        results_featjoint.append(fuse_bespoke_kfold(space_feat_jointsel,train_wholeset_size))
    result_featjoint_df=ml.pd.DataFrame(results_featjoint)
    result_featjoint_df.plot(x='train_size',y='fusion_mean_acc',style='o',title='feat level joint select')
    result_featjoint_df.plot(x='train_size',y='fusion_mean_acc',style='o',title='feat level joint select',ylim=(0,1))
    
    
    
    results_hierarch=[]
    for train_wholeset_size in train_sizes:
        
        space_hierarch=fuse.setup_search_space('hierarchical',include_svm=True)
        space_hierarch.update({'fusion_alg':'hierarchical',
                           'eeg':{'eeg_model_type':'kNN',
                                  'knn_k':12,},
                           'emg':{'emg_model_type':'SVM_PlattScale',
                                  'kernel':'rbf',
                                  #'svm_C':4.2807, 
                                  'svm_C':4.28074802128932,
                                  #'gamma':0.0102,
                                  'gamma':0.010215957329744027,},
                           'stack_distros':True,
                           })
        space_hierarch.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
                               
        results_hierarch.append(fuse_bespoke_kfold(space_hierarch,train_wholeset_size))
    result_hierarch_df=ml.pd.DataFrame(results_hierarch)
    result_hierarch_df.plot(x='train_size',y='fusion_mean_acc',style='o',title='hierarchical')
    result_hierarch_df.plot(x='train_size',y='fusion_mean_acc',style='o',title='hierarchical',ylim=(0,1))
    
    
    
    results_inv_hierarch=[]
    for train_wholeset_size in train_sizes:
        
        space_inv_hierarch=fuse.setup_search_space('hierarchical_inv',include_svm=True)
        space_inv_hierarch.update({'fusion_alg':'hierarchical_inv',
                               'eeg':{'eeg_model_type':'RF',
                                  'n_trees':70,
                                  'max_depth':5,},
                           'emg':{'emg_model_type':'SVM_PlattScale',
                                  'kernel':'rbf',
                                  #'svm_C':18.1983, 
                                  'svm_C':18.198340693618345,
                                  #'gamma':0.022,
                                  'gamma':0.02203231569864118,},
                           'stack_distros':True,
                           })
        space_inv_hierarch.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
                               
        results_inv_hierarch.append(fuse_bespoke_kfold(space_inv_hierarch,train_wholeset_size))
    result_inv_hierarch_df=ml.pd.DataFrame(results_inv_hierarch)
    result_inv_hierarch_df.plot(x='train_size',y='fusion_mean_acc',style='o',title='inverse hierarchical')
    result_inv_hierarch_df.plot(x='train_size',y='fusion_mean_acc',style='o',title='inverse hierarchical',ylim=(0,1))