# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 18:06:17 2023

@author: pritcham
"""
import time
import testFusion as fuse
import handleML as ml
import handleFeats as feats
import params
import numpy as np
import pandas as pd
import statistics as stats
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

def update_chosen_params(space,arch):
    paramdict={
            'just_emg':{'fusion_alg':'just_emg',
                        'emg':{'emg_model_type':'SVM_PlattScale',
                           'kernel':'rbf',
                           'svm_C':4.172505640055673,
                           'gamma':0.012556011834910268,
                           },
                      },
            'just_eeg':{'fusion_alg':'just_eeg',
                        'eeg':{'eeg_model_type':'LDA',
                           'LDA_solver':'lsqr',
                           'shrinkage':0.037969462143491395,
                           },
                      },
            'decision':{'fusion_alg':'highest_conf',
                      #  'ldafuse':{'LDA_solver':'eigen',
                       #            'shrinkage':0.21678705982755336,},
                        'eeg':{'eeg_model_type':'RF',
                               'n_trees':85,
                               'max_depth':5,},
                        'emg':{'emg_model_type':'SVM_PlattScale',
                               'kernel':'rbf',
                               'svm_C':98.91885185586297,
                               'gamma':0.013119782396855456,},
                        'stack_distros':True,
                        },
            'feat_sep':{'fusion_alg':'featlevel',
                        'featfuse':
                            {'featfuse_model_type':'LDA',
                             'LDA_solver':'svd',
                             'shrinkage':0.6653418849680925,
                             },
                        'featfuse_sel_feats_together':False,
                        },
            'feat_join':{'fusion_alg':'featlevel',
                         'featfuse':
                            {'featfuse_model_type':'LDA',
                             'LDA_solver':'svd',
                             'shrinkage':0.5048542532123359,
                             },
                        'featfuse_sel_feats_together':True,
                        },
            'hierarchical':{'fusion_alg':'hierarchical',
                           'eeg':{'eeg_model_type':'QDA',
                                  'regularisation':0.4558963480892469,},
                           'emg':{'emg_model_type':'SVM_PlattScale',
                                  'kernel':'rbf',
                                  'svm_C':19.403739187394663,
                                  'gamma':0.013797650887036847,},
                           'stack_distros':True,
                           },
            'hierarchical_inv':{'fusion_alg':'hierarchical_inv',
                           'eeg':{'eeg_model_type':'RF',
                                  'n_trees':75,
                                  'max_depth':5,},
                           'emg':{'emg_model_type':'QDA',
                                  'regularisation':0.3324563281128364,},
                           'stack_distros':True,
                           },
        }
    space.update(paramdict[arch])
    return space

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
    

    emg_ppt = emg_set
    eeg_ppt = eeg_set
    
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
                
    ppt_acc=(stats.mean(accs))
    ppt_emg=(stats.mean(emg_accs))
    ppt_eeg=(stats.mean(eeg_accs))
    ppt_train_acc=(stats.mean(train_accs))
    ppt_duration=(stats.mean(ppt_duration))

    return {
        'loss': 1-ppt_acc,
        'acc':ppt_acc,
        'emg_acc':ppt_emg,
        'eeg_acc':ppt_eeg,
        'train_acc':ppt_train_acc,
        'train_size':train_wholeset_size,
        'elapsed_time':ppt_duration,
        #'fusion_alg':args['fusion_alg'],
        #'ppt':('ppt'+str(int(emg_ppt['ID_pptID'].iloc[0])-1)),
        }


if __name__ == '__main__':

    ppt1={'emg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\emg_holdout_ppt1.csv",
          'eeg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\eeg_holdout_ppt1.csv",
          'ppt ID':'1'}
    ppt6={'emg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\emg_holdout_ppt6.csv",
          'eeg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\eeg_holdout_ppt6.csv",
          'ppt ID':'6'}
    ppt11={'emg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\emg_holdout_ppt11.csv",
          'eeg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\eeg_holdout_ppt11.csv",
          'ppt ID':'11'}
    ppt16={'emg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\emg_holdout_ppt16.csv",
          'eeg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\eeg_holdout_ppt16.csv",
          'ppt ID':'16'}
    ppt21={'emg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\emg_holdout_ppt21.csv",
          'eeg_path':r"H:\Jeong11tasks_data\final_dataset\holdout\eeg_holdout_ppt21.csv",
          'ppt ID':'21'}
    
    holdout_ppts=[ppt1,ppt6,ppt11,ppt16,ppt21]
    
    
    train_sizes1=0.77-np.geomspace(0.1,0.2,5)[::-1]
    train_sizes_mid=np.linspace(0.2925,0.57,4)
    train_sizes2=np.geomspace(0.02,0.2,10)
    train_sizes=np.unique(np.concatenate((train_sizes1,train_sizes_mid,train_sizes2)))
    #fuse.plt.plot(train_sizes,np.zeros(len(train_sizes))+1,marker='o');fuse.plt.show()
    
    gen_accs=pd.read_csv(r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\GeneralistOnHoldout\accuracies_withEEG.csv",index_col=0,header=0)
    gen_accs.loc[gen_accs['arch']=='hierarch','arch']='hierarchical'
    gen_accs.loc[gen_accs['arch']=='inv_hierarch','arch']='hierarchical_inv'
    
    archs=['just_emg','just_eeg','decision','feat_sep','feat_join','hierarchical','hierarchical_inv']
    #archs=['just_emg','feat_join']
    
    all_results=[]
    for ppt in holdout_ppts:
        emg=pd.read_csv(ppt['emg_path'],delimiter=',')
        eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
        emg,eeg=fuse.balance_set(emg,eeg)
       
        for arch in archs:
            space=fuse.setup_search_space(arch,include_svm=True)
            
            space=update_chosen_params(space,arch)
            
            space.update({'emg_set':emg,'eeg_set':eeg,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
            
            results=[]
            for train_wholeset_size in train_sizes:
                results.append(fuse_bespoke_kfold(space,train_wholeset_size))
            result_df=ml.pd.DataFrame(results)
            
            all_results.append({'arch':arch,
                                 'ppt':'ppt'+ppt['ppt ID'],
                                 'results':result_df})
    
    all_results_df=ml.pd.DataFrame(all_results)
    colours=['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
    for arch in archs:
        _,ax=plt.subplots()
        for idx, res in all_results_df[all_results_df['arch']==arch].iterrows():
            res['results'].plot(x='train_size',y='acc',ylim=(0,1),ax=ax,color=colours[idx],label=res['ppt'])
            genscore=gen_accs[(gen_accs['arch']==arch)&(gen_accs['ppt']==res['ppt'])]['fusion_acc'].values[0]
            ax.axhline(y=genscore,linestyle='--',color=colours[idx])
     #   ax.legend([res['ppt'] for _,res in all_results_df[all_results_df['arch']==arch].iterrows()])
        ax.set_title(arch)
        
   # result_df.plot(x='train_size',y='acc',title=(arch+' ppt '+ppt['ppt ID']),ylim=(0,1),ax=ax)
   # genscore=gen_accs[(gen_accs['arch']==arch)&(gen_accs['ppt']==('ppt'+ppt['ppt ID']))]['fusion_acc'].values[0]
   # ax.axhline(y=genscore,linestyle='--')
            
        
    raise ValueError('end')
    
    
    
    '''--------------------------------'''