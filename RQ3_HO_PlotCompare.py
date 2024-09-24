# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:53:04 2023

@author: pritcham
"""
from RQ2_opt import get_ppt_split_flexi, balance_set, fusion_SVM, fusion_LDA, fusion_RF, train_models_opt, refactor_synced_predict, classes_from_preds, setup_search_space, inspect_set_balance
import os
import sys
import numpy as np
import statistics as stats
import handleDatawrangle as wrangle
import handleFeats as feats
import handleML as ml
import handleComposeDataset as comp
import handleTrainTestPipeline as tt
import handleFusion as fusion
import params
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, log_loss, confusion_matrix, ConfusionMatrixDisplay #plot_confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.patches import Ellipse
from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, Trials
from hyperopt.pyll import scope, stochastic
import time
import pandas as pd
import pickle as pickle
from copy import deepcopy





gen_dev_accs={2: 0.76625, 3: 0.68875, 4: 0.7879166666666667, 5: 0.77875, 7: 0.81, 8: 0.745, 9: 0.6504166666666666,
              10: 0.6991666666666667, 12: 0.70375, 13: 0.5275, 14: 0.6008333333333333, 15: 0.65875,
              17: 0.8183333333333334, 18: 0.74875, 19: 0.7379166666666667, 20: 0.7408333333333333,
              22: 0.7375, 23: 0.6825, 24: 0.8375, 25: 0.7395833333333334}


if __name__ == '__main__':
    
    plt.rcParams['figure.dpi']=150
    
    summary_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\rq3HO_all_scores.xlsx"
    summary_scores=pd.read_excel(summary_res_path,index_col=0)
    
    xfer_gen=summary_scores[summary_scores['system']=='xfer_gen']
    xfer_prior=summary_scores[summary_scores['system']=='xfer_prior']
    
    prior_pretrain=summary_scores[summary_scores['system']=='prior_pretrain']
    
    within_session=summary_scores[summary_scores['system']=='within_session']
    within_opt_prior=summary_scores[summary_scores['system']=='within_opt_prior']
    
    
    nGest=4
    nRepsPerGest=50
    nInstancePerGest=4
    testset_size=0.33
    
    xfer_gen['calib_level_wholegests']=xfer_gen['calib_level']*(1-testset_size)*nGest*nRepsPerGest
    xfer_prior['calib_level_wholegests']=xfer_prior['calib_level']*(1-testset_size)*nGest*nRepsPerGest
    within_session['calib_level_wholegests']=within_session['calib_level']*(1-testset_size)*nGest*nRepsPerGest
    within_opt_prior['calib_level_wholegests']=within_opt_prior['calib_level']*(1-testset_size)*nGest*nRepsPerGest
    
    
    
    generalist_HO_featfus=[0.66667,0.85227,0.77652,0.70455,0.73106]
    #this was the winner Generalist, but then testing on only 1/3rd of session3 data to be consistent with others
    mean_gen_HO=np.mean(generalist_HO_featfus)
    std_gen_HO=np.std(generalist_HO_featfus)
    
    
    
    
    fig=plt.figure()
    ax=fig.add_axes((0.0,0.25,0.8,0.6))   
    
    
    within_session_agg=within_session.groupby(['calib_level_wholegests'])['acc'].agg(['mean','std']).reset_index()
    within_session_agg=within_session_agg.round({'calib_level_wholegests':5})
    within_session_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session\nlearning',color='tab:blue')#,yerr='std',capsize=2)
    
    
    xfer_prior_agg=xfer_prior.groupby(['calib_level_wholegests'])['acc'].agg(['mean','std']).reset_index()
    xfer_prior_agg=xfer_prior_agg.round({'calib_level_wholegests':5})
    xfer_prior_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Transfer from\nprior user data\nwith static\nconfiguration',color='tab:red')#,yerr='std',capsize=2)
    
    
    xfer_gen_agg=xfer_gen.groupby(['calib_level_wholegests'])['acc'].agg(['mean','std']).reset_index()
    xfer_gen_agg=xfer_gen_agg.round({'calib_level_wholegests':5})
  #  xfer_gen_agg.pivot(index='calib_level',
  #                   columns='ppt',
  #                   values='mean').
    xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Transfer from\nGeneralist',color='tab:brown')#,yerr='std',capsize=2)
                                        # yerr=scores_agg.pivot(index='calib_level_wholegests',
                                        #                       columns='subject id',values='std'))
    
    
    
    within_opt_prior_agg=within_opt_prior.groupby(['calib_level_wholegests'])['acc'].agg(['mean','std']).reset_index()
    within_opt_prior_agg=within_opt_prior_agg.round({'calib_level_wholegests':5})
    within_opt_prior_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session\ntraining (system\nconfig optimised\non downsampled\nprior user data)',color='tab:olive')#,yerr='std',capsize=2)
    
    plt.title('Mean accuracies over Holdout subjects\non reserved 33% of Session 3 data (66 gestures)',loc='left')
    ax.set_xlabel('# Session 3 gestures used for learning')
    ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
    
    plt.axhline(y=prior_pretrain['acc'].agg('mean'),label='Pretrained on\nprior user data',linestyle='--',color='black')
#    plt.axhline(y=0.841465,label='RQ2 HO Besp\nNot session-split',linestyle='-.',color='pink')
    #std dev for RQ2 Besp is 0.0643496, not sure if thats Dev or HO
    plt.axhline(y=mean_gen_HO,label='Generalist',linestyle='-.',color='gray')
   # plt.axhline(y=mean_gen_HO+std_gen_HO,linestyle=':',color='gray')
   # plt.axhline(y=mean_gen_HO-std_gen_HO,linestyle=':',color='gray')
    
    ax.legend(title='Approach',loc='center left',bbox_to_anchor=(1,0.5))
    ax.set_ylim(0.4,1.0)    
     
    axTime=fig.add_axes((0.0,0.1,0.8,0.0))
    axTime.yaxis.set_visible(False)
    axTime.set_xticks(ax.get_xticks())
    def tick_function(X):
        #V = 1/(1+X)
        V=(X*3)/60
        return ["%.1f" % z for z in V]
    axTime.set_xticklabels(tick_function(ax.get_xticks()))
    axTime.set_xlim(ax.get_xlim())
    axTime.set_xlabel("Minimum session-specific recording time (minutes)")
    #https://stackoverflow.com/questions/31803817/how-to-add-second-x-axis-at-the-bottom-of-the-first-one-in-matplotlib
    
    plt.show()
    
    
    
    
    fig=plt.figure()
    ax=fig.add_axes((0.0,0.15,0.8,0.8))   
    
    
    within_session_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session\nlearning',color='tab:blue')#,yerr='std',capsize=2)
    
    xfer_prior_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Transfer from\nprior user data\nwith static\nconfiguration',color='tab:red')#,yerr='std',capsize=2)
    
    xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Transfer from\nGeneralist',color='tab:brown')#,yerr='std',capsize=2)
    
    within_opt_prior_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Within-session\ntraining (system\nconfig optimised\non downsampled\nprior user data)',color='tab:olive')#,yerr='std',capsize=2)
    
    plt.title('Mean accuracies of trialled systems over Holdout subjects\non reserved 33% of Session 3 data (66 gestures)',loc='left')
    ax.set_xlabel('# Session 3 gestures used for learning')
    ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
    
    plt.axhline(y=prior_pretrain['acc'].agg('mean'),label='Pretrained on\nprior user data',linestyle='--',color='black')
#    plt.axhline(y=0.841465,label='RQ2 HO Besp\nNot session-split',linestyle='-.',color='pink')
    #std dev for RQ2 Besp is 0.0643496, not sure if thats Dev or HO
    plt.axhline(y=mean_gen_HO,label='Generalist',linestyle='-.',color='gray')
   # plt.axhline(y=mean_gen_HO+std_gen_HO,linestyle=':',color='gray')
   # plt.axhline(y=mean_gen_HO-std_gen_HO,linestyle=':',color='gray')
    
    ax.legend(title='Approach',loc='center left',bbox_to_anchor=(1,0.5))
    ax.set_ylim(0.4,1.0)    
     
    axTime=fig.add_axes((0.0,0.0,0.8,0.0))
    axTime.yaxis.set_visible(False)
    axTime.set_xticks(ax.get_xticks())
    def tick_function(X):
        #V = 1/(1+X)
        V=(X*3)/60
        return ["%.1f" % z for z in V]
    axTime.set_xticklabels(tick_function(ax.get_xticks()))
    axTime.set_xlim(ax.get_xlim())
    axTime.set_xlabel("Minimum session-specific recording time (minutes)")
    #https://stackoverflow.com/questions/31803817/how-to-add-second-x-axis-at-the-bottom-of-the-first-one-in-matplotlib
    
    plt.show()
    
      
    
    raise
    
    
    fig,ax=plt.subplots();
    xfer_gen_agg.plot(y='mean',x='calib_level',kind='line',ax=ax,rot=0,label='xfer_gen',color='tab:brown',yerr='std',capsize=2)
  #  xfer_prior_agg.plot(y='mean',x='calib_level',kind='line',ax=ax,rot=0,label='xfer_prior',color='tab:red',yerr='std',capsize=2)
    
    for ppt in xfer_gen['ppt'].unique():
        xfer_gen[xfer_gen['ppt']==ppt].plot(y='acc',x='calib_level',kind='line',ax=ax,label=ppt)

    plt.title('Accuracy per HO on reserved 33% of session 3 (66 gests)')
    ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
    ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
    
    plt.axhline(y=prior_pretrain['acc'].agg('mean'),label='prior_pretrain',linestyle='--',color='black')
    #plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
    plt.axhline(y=mean_gen_HO,label='Generalist',linestyle='-.',color='gray')
    #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
    ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
    plt.show()
    
    
    fig,ax=plt.subplots();
    xfer_prior_agg.plot(y='mean',x='calib_level',kind='line',ax=ax,rot=0,label='xfer_prior',color='tab:brown',yerr='std',capsize=2)
    for ppt in xfer_prior['ppt'].unique():
        xfer_prior[xfer_prior['ppt']==ppt].plot(y='acc',x='calib_level',kind='line',ax=ax,label=ppt)

    plt.title('Accuracy per HO on reserved 33% of session 3 (66 gests)')
    ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
    ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
    
    plt.axhline(y=prior_pretrain['acc'].agg('mean'),label='prior_pretrain',linestyle='--',color='black')
    #plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
    plt.axhline(y=mean_gen_HO,label='Generalist',linestyle='-.',color='gray')
    #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
    ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
    plt.show()
    
    
    fig,ax=plt.subplots();
    within_session_agg.plot(y='mean',x='calib_level',kind='line',ax=ax,rot=0,label='within_session',color='tab:brown',yerr='std',capsize=2)
    for ppt in within_session['ppt'].unique():
        within_session[within_session['ppt']==ppt].plot(y='acc',x='calib_level',kind='line',ax=ax,label=ppt)

    plt.title('Accuracy per HO on reserved 33% of session 3 (66 gests)')
    ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
    ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
    
    plt.axhline(y=prior_pretrain['acc'].agg('mean'),label='prior_pretrain',linestyle='--',color='black')
    #plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
    plt.axhline(y=mean_gen_HO,label='Generalist',linestyle='-.',color='gray')
    #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
    ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
    plt.show()
    
    
    fig,ax=plt.subplots();
    within_opt_prior_agg.plot(y='mean',x='calib_level',kind='line',ax=ax,rot=0,label='within_opt_prior',color='tab:brown',yerr='std',capsize=2)
    for ppt in within_opt_prior['ppt'].unique():
        within_opt_prior[within_opt_prior['ppt']==ppt].plot(y='acc',x='calib_level',kind='line',ax=ax,label=ppt)

    plt.title('Accuracy per HO on reserved 33% of session 3 (66 gests)')
    ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
    ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
    
    plt.axhline(y=prior_pretrain['acc'].agg('mean'),label='prior_pretrain',linestyle='--',color='black')
    #plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
    plt.axhline(y=mean_gen_HO,label='Generalist',linestyle='-.',color='gray')
    #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
    ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
    plt.show()
    
    raise
    
    
    
    
    plot_results=True
    load_res_path=None
    #load_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\D1_Warmstart_final_resMinimal - Copy.csv"
    
    #this is <warmstart from 1+2 train, no further opt>
    # ie Xfer No Opt
   # load_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\D2_Warmstart_NoOpt_final_resMinimal - Copy.csv"
    #RF was broken, fixed is below
    load_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\D2a_WarmNoOpt_FixRF_final_resMinimal - Copy.csv"
    
    #load_xfer_opt_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\D1_1_Warmstart_NoAppend_final_resMinimal - Copy.csv"
    #RF was broken, fixed is below
    load_xfer_opt_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\D1a_Warmstart_FixRF_final_resMinimal - Copy.csv"
    
    load_sessiononly_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\C1_Session3onlyfinal_resMinimal - Copy.csv"
    load_aug_unadjusted=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\B1_AugPipelinefinal_resMinimal - Copy.csv"
    
    load_aug_res_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\B1_1_AugAdjustedSplit_final_resMinimal - Copy.csv"
    
    load_withinSesh_noOpt=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\B2_WithinSession_noCal_final_resMinimal.csv"
    
    #load_warm_from_Gen=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\E2_Warmstart_fromGen_samesize_final_resMinimal.csv"
    #RF was broken, fixed is below
    load_warm_from_Gen=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\E2a_WarmFromGen_FixRF_final_resMinimal.csv"
    
    
    
    within_opt_2_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\B2a_WithinSession_optFor2_final_resMinimal.csv"
    within_opt_both_downsample_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\B2b_WithinSession_optForHalf_final_resMinimal.csv"
    train_both_baseline_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\A2_both1and2final_resMinimal.csv"
    train_2_baseline_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\A1b_session2final_resMinimal.csv"
    train_both_downsample_baseline_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\A2a_bothNoExtraData_final_resMinimal.csv"
    withinTrain_topupOpt_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ3\C2_WithinSession_TopupOptfinal_resMinimal - Copy.csv"
    
    

    
    testset_size = 0.33
        
 
    

    scores_minimal=pd.read_csv(load_res_path,index_col=0)        
    
    if plot_results:
        scores_aug_minimal=pd.read_csv(load_aug_res_path,index_col=0)
        scores_sessiononly_minimal=pd.read_csv(load_sessiononly_res_path,index_col=0)
        scores_xfer_opt=pd.read_csv(load_xfer_opt_path,index_col=0)
        scores_session_noOpt=pd.read_csv(load_withinSesh_noOpt,index_col=0)
        scores_XferGen=pd.read_csv(load_warm_from_Gen,index_col=0)
        
       # scores_aug_unadjusted=pd.read_csv(load_aug_unadjusted,index_col=0)
        
        
        scores_minimal=scores_minimal.round({'augment_scale':5})
        
        plt.rcParams['figure.dpi'] = 150 # DEFAULT IS 100
           
        
        nGest=4
        nRepsPerGest=50
        nInstancePerGest=4
        trainsplitSize=2/3
        scores_minimal['calib_level_instances']=scores_minimal['calib_level']*(1-testset_size)*nGest*nRepsPerGest*nInstancePerGest
        scores_minimal['calib_level_wholegests']=scores_minimal['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        scores_minimal['calib_level_pergest']=scores_minimal['calib_level']*(1-testset_size)*nRepsPerGest
        
                
        scores_aug_minimal['calib_level_wholegests']=scores_aug_minimal['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        scores_sessiononly_minimal['calib_level_wholegests']=scores_sessiononly_minimal['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        scores_xfer_opt['calib_level_wholegests']=scores_xfer_opt['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        scores_session_noOpt['calib_level_wholegests']=scores_session_noOpt['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        scores_XferGen['calib_level_wholegests']=scores_XferGen['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        
        
        rq2_bespoke_ref_path=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\RQ2\D1e_NoRolloff_Stablefinal_resMinimal - Copy.csv"
        scores_rq2=pd.read_csv(rq2_bespoke_ref_path,index_col=0)  
        scores_rq2['augscale_wholegests']=np.around(scores_rq2['augment_scale']*19*nGest*nRepsPerGest).astype(int)
        scores_rq2['trainAmnt_wholegests']=np.around(scores_rq2['rolloff_factor']*trainsplitSize*nGest*nRepsPerGest).astype(int)
        
        noAugRQ2=scores_rq2[scores_rq2['augscale_wholegests']==0]
        noAugRQ2=noAugRQ2.groupby(['trainAmnt_wholegests','augscale_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        
        

        
        if 0:
            fig,ax=plt.subplots();
            scores_agg=scores_minimal.groupby(['subject id','calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
            scores_agg=scores_agg.round({'calib_level_wholegests':5})
            scores_agg.pivot(index='calib_level_wholegests',
                             columns='subject id',
                             values='mean').plot(kind='bar',ax=ax,rot=0)#,capsize=2,width=0.8,
                                                # yerr=scores_agg.pivot(index='calib_level_wholegests',
                                                #                       columns='subject id',values='std'))
            '''only relevant yerr here would be if i got multiple shots per ppt - which would be nice'''
            ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
            plt.title('Accuracy per subject on reserved 33% of session 3 (66 gests)')
            ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
            ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
            
            plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
            plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
            plt.axhline(y=0.7475,label='Train both\n(no cal) avg',linestyle='--',color='black')
            ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
            #ax.set_ylim(0.3,0.95)
            plt.show()
        
        
        fig,ax=plt.subplots();
        scores_agg=scores_minimal.groupby(['subject id','calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_agg=scores_agg.round({'calib_level_wholegests':5})
        scores_agg.pivot(index='calib_level_wholegests',
                         columns='subject id',
                         values='mean').plot(kind='line',ax=ax,rot=0)#,capsize=2,width=0.8,
                                            # yerr=scores_agg.pivot(index='calib_level_wholegests',
                                            #                       columns='subject id',values='std'))
        '''only relevant yerr here would be if i got multiple shots per ppt - which would be nice'''
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Accuracy per subject on reserved 33% of session 3 (66 gests)')
        ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        #plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0],label='RQ2 Bespoke\n(Not session-split)',linestyle='--',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0]+noAugRQ2['std'][0],linestyle=':',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0]-noAugRQ2['std'][0],linestyle=':',color='pink')
        plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        plt.axhline(y=0.7475,label='Train both\n(no cal) avg',linestyle='--',color='black')
        ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        plt.show()
        
        
        '''
        fig,ax=plt.subplots();
        scores_agg=scores_minimal.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_agg=scores_agg.round({'calib_level_wholegests':5})
        scores_agg.plot(y='mean',x='calib_level_wholegests',kind='bar',ax=ax,rot=0,yerr='std',capsize=5)
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies over subjects on reserved 33% of session 3 (66 gests)')
        ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        #plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        #plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        plt.axhline(y=0.7475,label='Train both\n(no cal) avg',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
        plt.show()
        '''
        
        
        
        
        
        scores_agg=scores_minimal.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_agg=scores_agg.round({'calib_level_wholegests':5})
        
        scores_xfer_opt_agg=scores_xfer_opt.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_xfer_opt_agg=scores_xfer_opt_agg.round({'calib_level_wholegests':5})
        
        scores_aug_agg=scores_aug_minimal.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_aug_agg=scores_aug_agg.round({'calib_level_wholegests':5})
        
        scores_sessiononly_agg=scores_sessiononly_minimal.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_sessiononly_agg=scores_sessiononly_agg.round({'calib_level_wholegests':5})
        
        scores_sessionNoOpt_agg=scores_session_noOpt.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_sessionNoOpt_agg=scores_sessionNoOpt_agg.round({'calib_level_wholegests':5})
        
        scores_xfer_gen_agg=scores_XferGen.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        scores_xfer_gen_agg=scores_xfer_gen_agg.round({'calib_level_wholegests':5})
        
        
        fig,ax=plt.subplots();
        
        
        if 0:
            scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only',yerr='std',capsize=5)
            scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented',yerr='std',capsize=5)
            scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\ntrained on 1&2',yerr='std',capsize=5)
            scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nwithout opt',yerr='std',capsize=5)
            scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only\nwithout opt',yerr='std',capsize=5)
            scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist',yerr='std',capsize=5)
        else:
            scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only')
            scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented')
            scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\ntrained on 1&2')
            scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nwithout opt')
            scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only\nwithout opt')
            scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist')
                    
 #       scores_aug_unadjusted['calib_level_wholegests']=scores_aug_unadjusted['calib_level']*(1-testset_size)*nGest*nRepsPerGest
 #       aug_unadjust_agg=scores_aug_adjusted.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
 #       aug_unadjust_agg=aug_adjust_agg.round({'calib_level_wholegests':5})
 #       aug_unadjust_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented\nadjusted split')
                
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies over subjects on reserved 33% of session 3 (66 gests)')
        ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        #below is 869066 taken from RQ2's A1_FullBespoke (in "provis")
        # but MIGHT actually be 865025 if we used D1b_RolloffStable5trials (which we use for RQ2 Hyp1)
        # OR 866105 if we use D1a_AugStable_with_errorCalcs
      #  plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        # NOW we take it from RQ2 rolloffStable5trials, but corrected (previously 5 trials werent different)
        plt.axhline(y=noAugRQ2['mean'][0],label='RQ2 Bespoke\n(Not session-split)',linestyle='--',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0]+noAugRQ2['std'][0],linestyle=':',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0]-noAugRQ2['std'][0],linestyle=':',color='pink')
        #plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        #ax.set_ylim(0.4,0.9)
        plt.axhline(y=0.7475,label='Train 1&2\n(no cal) avg',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
        plt.show()
        
        
        
        
        '''** ALL **'''
        
        fig,ax=plt.subplots();
        
        within_opt_2=pd.read_csv(within_opt_2_path,index_col=0)
        within_opt_both_downsample=pd.read_csv(within_opt_both_downsample_path,index_col=0)
        train_both_baseline=pd.read_csv(train_both_baseline_path,index_col=0)
        train_2_baseline=pd.read_csv(train_2_baseline_path,index_col=0)
        train_both_downsample_baseline=pd.read_csv(train_both_downsample_baseline_path,index_col=0)
        topupOpt=pd.read_csv(withinTrain_topupOpt_path,index_col=0)
        
        
        within_opt_2['calib_level_wholegests']=within_opt_2['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        within_opt_both_downsample['calib_level_wholegests']=within_opt_both_downsample['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        train_both_baseline['calib_level_wholegests']=train_both_baseline['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        train_2_baseline['calib_level_wholegests']=train_2_baseline['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        train_both_downsample_baseline['calib_level_wholegests']=train_both_downsample_baseline['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        topupOpt['calib_level_wholegests']=topupOpt['calib_level']*(1-testset_size)*nGest*nRepsPerGest
        
        
        within_opt_2_agg=within_opt_2.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        within_opt_both_downsample_agg=within_opt_both_downsample.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        topupOpt_agg=topupOpt.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
        
        train_both_baseline_score=np.mean(train_both_baseline['fusion_acc'])
        train_2_baseline_score=np.mean(train_2_baseline['fusion_acc'])
        train_both_downsample_baseline_score=np.mean(train_both_downsample_baseline['fusion_acc'])
        
        
        if 0:
            scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only',yerr='std',capsize=5)
            scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented',yerr='std',capsize=5)
            scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\ntrained on 1&2',yerr='std',capsize=5)
            scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nwithout opt',yerr='std',capsize=5)
            scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only\nwithout opt',yerr='std',capsize=5)
            scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist',yerr='std',capsize=5)
        else:
            scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only')
            scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented')
            scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom 1 + 2 incl\nopt for transfer')
            scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom static 1 + 2\nno further opt')
            scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 train\nOptimised for 1+2')
            scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist')
            within_opt_2_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 train\nOptimised for 2')
            within_opt_both_downsample_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 train\nOptimised for 1+2\n(downsampled to half)')
            topupOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 train\nOptimised for 3 topped\nup by 1+2 to 200 total')
            plt.axhline(y=train_both_baseline_score,label='Train 1 + 2',linestyle='--',color='tab:orange')
            plt.axhline(y=train_2_baseline_score,label='Train session 2',linestyle='--',color='tab:green')
            plt.axhline(y=train_both_downsample_baseline_score,label='Train 1 + 2\n(downsampled to half)',linestyle='--',color='tab:red')
                    
 #       scores_aug_unadjusted['calib_level_wholegests']=scores_aug_unadjusted['calib_level']*(1-testset_size)*nGest*nRepsPerGest
 #       aug_unadjust_agg=scores_aug_adjusted.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
 #       aug_unadjust_agg=aug_adjust_agg.round({'calib_level_wholegests':5})
 #       aug_unadjust_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented\nadjusted split')
                
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('ALL Mean accuracies over Development subjects on reserved 33% of session 3 (66 gests)',loc='left')
        ax.set_xlabel('# Session 3 gestures (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        #below is 869066 taken from RQ2's A1_FullBespoke (in "provis")
        # but MIGHT actually be 865025 if we used D1b_RolloffStable5trials (which we use for RQ2 Hyp1)
        # OR 866105 if we use D1a_AugStable_with_errorCalcs
      #  plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        # NOW we take it from RQ2 rolloffStable5trials, but corrected (previously 5 trials werent different)
        plt.axhline(y=noAugRQ2['mean'][0],label='RQ2 Bespoke\n(Not session-split)',linestyle='--',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0]+noAugRQ2['std'][0],linestyle=':',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0]-noAugRQ2['std'][0],linestyle=':',color='pink')
        plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        #ax.set_ylim(0.4,0.9)
       # plt.axhline(y=0.7475,label='Train 1&2\n(no cal) avg',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        plt.show()
        
        
        
        
        
        '''** INTERESTING **'''
        
        fig,ax=plt.subplots();
        
        
        if 0:
            scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only',yerr='std',capsize=5)
            scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented',yerr='std',capsize=5)
            scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\ntrained on 1&2',yerr='std',capsize=5)
            scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nwithout opt',yerr='std',capsize=5)
            scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only\nwithout opt',yerr='std',capsize=5)
            scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist',yerr='std',capsize=5)
        else:
            scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 within-session')
            scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmenting\nsessions 1 + 2')
           # scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom 1 + 2 incl\nopt for transfer')
            scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom static 1 + 2',c='tab:red')
            scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist',c='tab:brown')
            scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 trained\noptimised for 1+2',c='tab:purple')
         #   within_opt_2_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 train\nOptimised for 2')
            within_opt_both_downsample_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 trained\noptimised for 1+2\n(downsampled to half)',c='tab:gray')
          #  topupOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 train\nOptimised for 3 topped\nup by 1+2 to 200 total')
            plt.axhline(y=train_both_baseline_score,label='Optimised & Trained \non sessions 1 + 2',linestyle='--',color='black')
         #   plt.axhline(y=train_2_baseline_score,label='Train session 2',linestyle='--',color='tab:green')
         #   plt.axhline(y=train_both_downsample_baseline_score,label='Train 1 + 2\n(downsampled to half)',linestyle='--',color='tab:red')
                    
 #       scores_aug_unadjusted['calib_level_wholegests']=scores_aug_unadjusted['calib_level']*(1-testset_size)*nGest*nRepsPerGest
 #       aug_unadjust_agg=scores_aug_adjusted.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
 #       aug_unadjust_agg=aug_adjust_agg.round({'calib_level_wholegests':5})
 #       aug_unadjust_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented\nadjusted split')
                
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('INTERESTING Mean accuracies over Development subjects\non reserved 33% of session 3 (66 gestures)',loc='left')
        ax.set_xlabel('# Session 3 gestures (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        #below is 869066 taken from RQ2's A1_FullBespoke (in "provis")
        # but MIGHT actually be 865025 if we used D1b_RolloffStable5trials (which we use for RQ2 Hyp1)
        # OR 866105 if we use D1a_AugStable_with_errorCalcs
      #  plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        # NOW we take it from RQ2 rolloffStable5trials, but corrected (previously 5 trials werent different)
        plt.axhline(y=noAugRQ2['mean'][0],label='RQ2 Bespoke\n(Not session-split)',linestyle='-.',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0]+noAugRQ2['std'][0],linestyle=':',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0]-noAugRQ2['std'][0],linestyle=':',color='pink')
        plt.axhline(y=0.723,label='RQ1 Generalist\n(Not session-split)',linestyle='-.',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        #ax.set_ylim(0.4,0.9)
        #plt.axhline(y=0.7475,label='Train 1&2\n(no cal) avg',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),ncol=1)
        plt.show()
        
        
        
        '''** VIABLE **'''
        
        fig,ax=plt.subplots();
        
        
        if 0:
            scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only',yerr='std',capsize=5)
            scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented',yerr='std',capsize=5)
            scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\ntrained on 1&2',yerr='std',capsize=5)
            scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nwithout opt',yerr='std',capsize=5)
            scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 only\nwithout opt',yerr='std',capsize=5)
            scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist',yerr='std',capsize=5)
        else:
            scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 within-session')
          #  scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmenting\nsessions 1 + 2')
           # scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom 1 + 2 incl\nopt for transfer')
            scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom static 1 + 2',c='tab:red')
            scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist',c='tab:brown')
         #   scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 trained\noptimised for 1+2',c='tab:purple')
         #   within_opt_2_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 train\nOptimised for 2')
            within_opt_both_downsample_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 trained\noptimised for 1+2\n(downsampled to half)',c='tab:gray')
          #  topupOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 train\nOptimised for 3 topped\nup by 1+2 to 200 total')
            plt.axhline(y=train_both_baseline_score,label='Optimised & Trained \non sessions 1 + 2',linestyle='--',color='black')
         #   plt.axhline(y=train_2_baseline_score,label='Train session 2',linestyle='--',color='tab:green')
         #   plt.axhline(y=train_both_downsample_baseline_score,label='Train 1 + 2\n(downsampled to half)',linestyle='--',color='tab:red')
                    
 #       scores_aug_unadjusted['calib_level_wholegests']=scores_aug_unadjusted['calib_level']*(1-testset_size)*nGest*nRepsPerGest
 #       aug_unadjust_agg=scores_aug_adjusted.groupby(['calib_level_wholegests'])['fusion_acc'].agg(['mean','std']).reset_index()
 #       aug_unadjust_agg=aug_adjust_agg.round({'calib_level_wholegests':5})
 #       aug_unadjust_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmented\nadjusted split')
                
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('VIABLE Mean accuracies over Development subjects\non reserved 33% of session 3 (66 gestures)',loc='left')
        ax.set_xlabel('# Session 3 gestures (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
        #below is 869066 taken from RQ2's A1_FullBespoke (in "provis")
        # but MIGHT actually be 865025 if we used D1b_RolloffStable5trials (which we use for RQ2 Hyp1)
        # OR 866105 if we use D1a_AugStable_with_errorCalcs
      #  plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        # NOW we take it from RQ2 rolloffStable5trials, but corrected (previously 5 trials werent different)
        plt.axhline(y=noAugRQ2['mean'][0],label='RQ2 Bespoke\n(Not session-split)',linestyle='-.',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0]+noAugRQ2['std'][0],linestyle=':',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0]-noAugRQ2['std'][0],linestyle=':',color='pink')
        plt.axhline(y=0.723,label='RQ1 Generalist\n(Not session-split)',linestyle='-.',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        #ax.set_ylim(0.4,0.9)
        #plt.axhline(y=0.7475,label='Train 1&2\n(no cal) avg',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),ncol=1)
        plt.show()
        
        
        
        
        
        
        
        
        
        '''
        fig,ax=plt.subplots();
        scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 within-session')
        scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmenting 1+2')
        scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\n(optimised for transfer)')
        scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\n(from naive 1+2 model)')
        
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies over subjects on reserved 33% of session 3 (66 gests)')
        ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
       # plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        #plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        #ax.set_ylim(0.3,0.95)
        plt.axhline(y=0.7475,label='Trained & opt for\n1+2 (no cal)',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
        plt.show()
        '''
        
        
        '''
        fig,ax=plt.subplots();
        scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 within-session')
        scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmenting 1+2')
        scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\n(optimised for transfer)')
        scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\n(from naive 1+2 model)')
        
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies over subjects on reserved 33% of session 3 (66 gests)')
        ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
       # plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        #plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        ax.set_ylim(0.6,0.9)
        #ax.set_xlim(0,135)
        plt.axhline(y=0.7475,label='Trained on & opt-ed for\n1+2 (no cal)',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
        plt.show()
        '''
        
        
        
        
        fig,ax=plt.subplots();
        scores_sessiononly_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 within-session')
        scores_aug_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 augmenting 1+2')
        scores_xfer_opt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\n(optimised for transfer)')
        scores_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\n(from naive 1+2 model)')
        scores_sessionNoOpt_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 within-session\nwithout opt')
        scores_xfer_gen_agg.plot(y='mean',x='calib_level_wholegests',kind='line',marker='.',ax=ax,rot=0,label='Session 3 transfer\nfrom generalist')
        
        ax.set_ylim(np.floor(scores_minimal['fusion_acc'].min()/0.05)*0.05,np.ceil(scores_minimal['fusion_acc'].max()/0.05)*0.05)
        plt.title('Mean accuracies over subjects on reserved 33% of session 3 (66 gests)')
        ax.set_xlabel('# Session 3 gestures calibrating (max 134)')
        ax.set_ylabel('Classification Accuracy')#' on reserved 33% (200) subject')
        
       # plt.axhline(y=0.86907,label='RQ2 Full Besp\nNot session-split!',linestyle='--',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0],label='RQ2 Bespoke\n(Not session-split)',linestyle='--',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0]+noAugRQ2['std'][0],linestyle=':',color='pink')
        plt.axhline(y=noAugRQ2['mean'][0]-noAugRQ2['std'][0],linestyle=':',color='pink')
        #plt.axhline(y=0.723,label='RQ1 Generalist\nNot session-split!',linestyle='--',color='gray')
        #ax.legend(title='Subject',loc='center left',bbox_to_anchor=(1,0.5),ncol=2)
        ax.set_ylim(0.6,0.9)
        #ax.set_xlim(0,135)
        plt.axhline(y=0.7475,label='Trained on & opt-ed for\n1+2 (no cal)',linestyle='--',color='black')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
        
        
        tPerGest=4
        nCalibTotal=134
        tTotal=nCalibTotal*tPerGest
        tTotal_mins = tTotal/60
        t_save_mins = 6
        t_save=t_save_mins*60
        nCalibSave=np.floor((t_save/tPerGest)/4)*4
        #https://stackoverflow.com/questions/14892619/annotating-dimensions-in-matplotlib
        #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axvspan.html
        #https://stackoverflow.com/questions/36423221/matplotlib-axvspan-solid-fill
        
        #plt.axhspan(ymin=np.max(scores_sessiononly_agg)['mean']-0.03,ymax=np.max(scores_sessiononly_agg)['mean'],
        accept_loss = 1
        maxacc=0.8532196970500001
        min_acceptable=0.8532196970500001-(accept_loss/100)
        plt.axhspan(ymin=min_acceptable,ymax=0.8532196970500001, #ymin=0.8232196970500001
                    xmin=0.175,xmax=132/138,linestyle=':',lw=0.5,alpha=0.5,color='gray')
        ax.annotate("",xy=(138*0.1375,min_acceptable*0.995),#could do just 0.125 with xycoords=('axes fraction','data'))
                    xytext=(138*0.1375,0.8532196970500001/0.999),
                    arrowprops=dict(arrowstyle='->'))
    #    ax.annotate("",xy=(138*0.125,0.8232196970500001),xytext=(138*0.125,0.8532196970500001),
    #                arrowprops=dict(arrowstyle='|-|'))
       # bbox=dict(fc="white", ec="none")
        ax.text(138*0.05,(min_acceptable+0.8532196970500001)/2,
                f"Acceptable\naccuracy\nloss: {accept_loss}%", ha="center", va="center",size='small')#, bbox=bbox)
        
        
        plt.axvspan(132-nCalibSave,132,ymin=0.15,ymax=np.max(scores_sessiononly_agg)['mean']*0.99,
                    linestyle=':',lw=0.5,alpha=0.5,color='gray')
        
        ax.annotate("",xy=(132/0.995,0.15),xycoords=('data','axes fraction'),
                    xytext=((132-nCalibSave)*0.985,0.15),textcoords=('data','axes fraction'),
                    arrowprops=dict(arrowstyle='<-'))
       # bbox=dict(fc="white", ec="none")
        ax.text(132-(nCalibSave*0.5),0.625,
                f"Desired time reduction: {t_save_mins} mins", ha="center", va="center",size='small')#, bbox=bbox)
        plt.show()
        
    


    
if 0:
    def load_results_obj(path):
        load_trials=pickle.load(open(path,'rb'))
        load_table=pd.DataFrame(load_trials.trials)
        load_table_readable=pd.concat(
            [pd.DataFrame(load_table['result'].tolist()),
             pd.DataFrame(pd.DataFrame(load_table['misc'].tolist())['vals'].values.tolist())],
            axis=1,join='outer')
        return load_trials,load_table,load_table_readable
    
    _,_,gen_results=load_results_obj(r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Generalist_20DevSet\Gen_feat_joint\trials_obj.p")
    gen_best=gen_results.iloc[76]
    gen_best_accs=gen_best['fusion_accs']
    gen_dev_accs=dict(zip(scores_minimal['subject id'].unique(),gen_best_accs))
    
    

    