# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:34:23 2023

@author: pritcham
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import testFusion as fuse
import scipy.stats as stats

def boxplot_param(df_in,param,target,ylower=0,yupper=1,showplot=True,xlabs=None,title=None,titleheight=0.98):
    fig,ax=plt.subplots()
    dataframe=df_in.copy()
    if isinstance(dataframe[param][0],list):
        dataframe[param]=dataframe[param].apply(lambda x: x[0])
    dataframe.boxplot(column=target,by=param,ax=ax,showmeans=True)
    ax.set_ylim(ylower,yupper)
    if xlabs is not None:
        #ax.set_xticks(np.arange(1,len(xlabs)+1),xlabs)
        ax.set_xticklabels(xlabs)
   # plt.suptitle('')
    ax.set_title('')
    if title is not None:
    #    ax.set_title(title)
        plt.suptitle(title,y=titleheight)
    if showplot:
        plt.show()
    return fig

def model_significance(df,target,param,winner,title='Mean accuracy per classifier in Bespoke EEG-Only optimisation'):
    
    per_param=boxplot_param(df,param,target,xlabs=models,
                               title=title)
    
    stat_test=df[[param,target]]
    stat_test[param]=[x[0] for x in stat_test[param]]
    stat_test[param]=[model_dict[x] for x in stat_test[param]]
    groups = [stat_test[stat_test[param] == group][target] for group in models]
    #sorting the above by bespoke_models instead of #stat_test['eeg model'].unique()] #so that order is preserved
    fstat, pvalue = stats.f_oneway(*groups)
    print('anova on all ',fstat, pvalue)
    #https://saturncloud.io/blog/anova-in-python-using-pandas-dataframe-with-statsmodels-or-scipy/#:~:text=ANOVA%20is%20a%20fundamental%20statistical,popular%20libraries%3A%20Statsmodels%20and%20Scipy.
    
    '''COPY DATA FROM STAT_TEST FOR R SCRIPT'''
    
    #tukey not in scipy 1.4.1
    #tukey=stats.tukey_hsd(*groups)
    #print(tukey)
    
    win_idx=next(key for key,value in model_dict.items() if value==winner)
    for i in range(len(groups)):
        if i==win_idx:
            continue
        if np.std(groups[win_idx]) - np.std(groups[i]) > 0.05:
            #use math.isclose() ?
            tscore,pval=stats.ttest_ind(groups[win_idx],groups[i],equal_var=False)
            print('Welch\'s t on ',model_dict[win_idx],' & ',model_dict[i],': ',tscore,pval)
        else:
           # tscore,pval=stats.ttest_ind(groups[win_idx],groups[i],equal_var=True)
           # print('Student\'s t on ',model_dict(win_idx),' & ',model_dict(i),': ',tscore,pval)
            '''recommendations that variance equality not be checked for
            https://bpspsychub.onlinelibrary.wiley.com/doi/abs/10.1348/000711004849222
            and Welchs used throughout instead, despite being worse at low sample size'''
            tscore,pval=stats.ttest_ind(groups[win_idx],groups[i],equal_var=False)
            print('Welch\'s t on ',model_dict[win_idx],' & ',model_dict[i],': ',tscore,pval)
            
    return per_param

def hierarch_topmodel(df,target,param,title='Mean accuracy per classifier in Bespoke Hierarchical optimisation',titleheight=0.98):
    
    per_param=boxplot_param(df,param,target,xlabs=models,
                               title=title,titleheight=titleheight)
    
    stat_test=df[[param,target]]
    stat_test[param]=[x[0] for x in stat_test[param]]
    stat_test[param]=[model_dict[x] for x in stat_test[param]]
    groups = [stat_test[stat_test[param] == group][target] for group in models]
    #sorting the above by bespoke_models instead of #stat_test['eeg model'].unique()] #so that order is preserved
    fstat, pvalue = stats.f_oneway(*groups)
    print('anova on all ',fstat, pvalue)
    #https://saturncloud.io/blog/anova-in-python-using-pandas-dataframe-with-statsmodels-or-scipy/#:~:text=ANOVA%20is%20a%20fundamental%20statistical,popular%20libraries%3A%20Statsmodels%20and%20Scipy.
    
    '''COPY DATA FROM STAT_TEST FOR R SCRIPT'''

    return per_param, stat_test

def hierarch_lowmodel(df,target,param,title='Mean accuracy per classifier in Bespoke Hierarchical optimisation',titleheight=0.98):
    
    per_param=boxplot_param(df,param,target,xlabs=models,
                               title=title,titleheight=titleheight)
    
    stat_test=df[[param,target]]
    stat_test[param]=[x[0] for x in stat_test[param]]
    stat_test[param]=[model_dict[x] for x in stat_test[param]]
    groups = [stat_test[stat_test[param] == group][target] for group in models]
    #sorting the above by bespoke_models instead of #stat_test['eeg model'].unique()] #so that order is preserved
    fstat, pvalue = stats.f_oneway(*groups)
    print('anova on all ',fstat, pvalue)
    #https://saturncloud.io/blog/anova-in-python-using-pandas-dataframe-with-statsmodels-or-scipy/#:~:text=ANOVA%20is%20a%20fundamental%20statistical,popular%20libraries%3A%20Statsmodels%20and%20Scipy.
    
    '''COPY DATA FROM STAT_TEST FOR R SCRIPT'''
    return per_param#, stat_test

def fusalg_significance(df,target,param,winner,title='Mean accuracy per fusion alg in Bespoke Decision-fusion optimisation'):
    labels=['Mean','3:1 EMG','3:1 EEG','Opt\nweighted','Max','SVM','LDA','RF']
    per_param=boxplot_param(df,param,target,xlabs=labels,
                               title=title)
    
    stat_test=df[[param,target]]
    blocks=df[[param,'emg model','eeg model',target]]
    stat_test[param]=[x[0] for x in stat_test[param]]
    stat_test[param]=[alg_dict[x] for x in stat_test[param]]
    
    blocks[param]=[x[0] for x in blocks[param]]
    blocks[param]=[alg_dict[x] for x in blocks[param]]
    blocks['emg model']=[x[0] for x in blocks['emg model']]
    blocks['emg model']=[model_dict[x] for x in blocks['emg model']]
    blocks['eeg model']=[x[0] for x in blocks['eeg model']]
    blocks['eeg model']=[model_dict[x] for x in blocks['eeg model']]
    
    groups = [stat_test[stat_test[param] == group][target] for group in algs]
    #sorting the above by bespoke_models instead of #stat_test['eeg model'].unique()] #so that order is preserved
    fstat, pvalue = stats.f_oneway(*groups)
    print('anova on all ',fstat, pvalue)
    #https://saturncloud.io/blog/anova-in-python-using-pandas-dataframe-with-statsmodels-or-scipy/#:~:text=ANOVA%20is%20a%20fundamental%20statistical,popular%20libraries%3A%20Statsmodels%20and%20Scipy.
    
    '''COPY DATA FROM STAT_TEST FOR R SCRIPT'''
    
    #tukey not in scipy 1.4.1
    #tukey=stats.tukey_hsd(*groups)
    #print(tukey)
            
    return per_param

            
def LDA_solver_significance(df,modelparam,resultparam,solverparam,shrinkageparam):
    df_subset=df[[modelparam,resultparam,solverparam,shrinkageparam]]
    df_subset[modelparam]=[x[0] for x in df_subset[modelparam]]
    df_subset=df_subset.loc[df_subset[modelparam]==2].reset_index(drop=True)
    df_subset[solverparam]=[x[0] for x in df_subset[solverparam]]
    df_subset[shrinkageparam]=[x[0] for x in df_subset[shrinkageparam]]
    
    #CAN COPY COLUMNS FROM DF_SUBSET FOR R SCRIPT
    
    solvers=['svd','lsqr','eigen']
    solver_dict=dict(zip(range(0,len(solvers)+1),solvers))
    
    solver_groups = [df_subset[df_subset[solverparam] == group][resultparam] for group in df_subset[solverparam].unique()]
    fstat, pvalue = stats.f_oneway(*solver_groups)
    print('anova on all ',fstat, pvalue)
    per_LDAsolver=boxplot_param(df_subset,solverparam,resultparam,xlabs=solvers,
                               title='Mean accuracy per LDA solver, ANOVA: f='+str(round(fstat,4))+', p='+str(round(pvalue,4)))
    
    print('below only looks for linear correlation, optimiser looks for local minimum')
    LDAs_perSolver=[df_subset[df_subset[solverparam]==group].reset_index(drop=True) for group in [0,1,2]]#df_subset[solverparam].unique()]
    df_subset_noSVD=df_subset.loc[df_subset[solverparam]!=0].reset_index(drop=True)
    pearsonR=stats.pearsonr(df_subset_noSVD[shrinkageparam],df_subset_noSVD[resultparam])
    spearmanR=stats.spearmanr(df_subset_noSVD[shrinkageparam],df_subset_noSVD[resultparam])
   # df_subset_noSVD.plot(x=shrinkageparam,y=resultparam,kind='scatter',
    #                    title='Accuracy vs LDA shrinkage, pearson coefficient = '+str((round(pearsonR[0],4),round(pearsonR[1],4))))
   # df_subset_noSVD.plot(x=shrinkageparam,y=resultparam,kind='scatter',
    #                    title='Accuracy vs LDA shrinkage, spearman rho = '+str((round(spearmanR[0],4),round(spearmanR[1],4))))
    df_subset_noSVD.plot(x=shrinkageparam,y=resultparam,kind='scatter',
                        title='Accuracy vs LDA shrinkage, pearson coefficient = '+str((round(pearsonR[0],4),round(pearsonR[1],4)))
                        +'\n'+f"{' '*28}"+' spearman rho = '+str((round(spearmanR[0],4),round(spearmanR[1],4))))
    
    pearsonR=stats.pearsonr(LDAs_perSolver[1][shrinkageparam],LDAs_perSolver[1][resultparam])
    spearmanR=stats.spearmanr(LDAs_perSolver[1][shrinkageparam],LDAs_perSolver[1][resultparam])
  #  LDAs_perSolver[1].plot(x=shrinkageparam,y=resultparam,kind='scatter',
   #                     title='Accuracy vs shrinkage, solver: '+str(solver_dict[1])+', pearson coefficient = '+str((round(pearsonR[0],4),round(pearsonR[1],4))))
    LDAs_perSolver[1].plot(x=shrinkageparam,y=resultparam,kind='scatter',
                        title='Accuracy vs shrinkage, solver: '+str(solver_dict[1])
                        +'\npearson coefficient = '+str((round(pearsonR[0],4),round(pearsonR[1],4)))
                        +'\nspearman rho = '+str((round(spearmanR[0],4),round(spearmanR[1],4))))
    
    pearsonR=stats.pearsonr(LDAs_perSolver[2][shrinkageparam],LDAs_perSolver[2][resultparam])
    spearmanR=stats.spearmanr(LDAs_perSolver[2][shrinkageparam],LDAs_perSolver[2][resultparam])
    #LDAs_perSolver[2].plot(x=shrinkageparam,y=resultparam,kind='scatter',
     #                   title='Accuracy vs shrinkage, solver: '+str(solver_dict[2])+', pearson coefficient = '+str((round(pearsonR[0],4),round(pearsonR[1],4))))
    LDAs_perSolver[2].plot(x=shrinkageparam,y=resultparam,kind='scatter',
                        title='Accuracy vs shrinkage, solver: '+str(solver_dict[2])
                        +'\npearson coefficient = '+str((round(pearsonR[0],4),round(pearsonR[1],4)))
                        +'\nspearman rho = '+str((round(spearmanR[0],4),round(spearmanR[1],4))))


def SVM_params_significance(df,modelparam,resultparam,Cparam,Gammaparam,trial=None):
    df_subset=df[[modelparam,resultparam,Cparam,Gammaparam]]
    df_subset[modelparam]=[x[0] for x in df_subset[modelparam]]
    df_subset=df_subset.loc[df_subset[modelparam]==5].reset_index(drop=True)
    df_subset[Cparam]=[x[0] for x in df_subset[Cparam]]
    df_subset[Gammaparam]=[x[0] for x in df_subset[Gammaparam]]
    
    pearsonR=stats.pearsonr(df_subset[Cparam],df_subset[resultparam])
    df_subset.plot(x=Cparam,y=resultparam,kind='scatter')#,logx=True)#ylim=(0.8,1),
    title=trial+'\nAccuracy vs C, pearson coefficient = '+(str((round(pearsonR[0],4),round(pearsonR[1],4))) if round(pearsonR[1],4)!=0 else '('+str(round(pearsonR[0],4))+', <0.0001)')
    plt.gcf().suptitle(title,y=0.995)
    
    pearsonR=stats.pearsonr(df_subset[Gammaparam],df_subset[resultparam])
    df_subset.plot(x=Gammaparam,y=resultparam,kind='scatter')#,logx=True)
    #title=trial+'\nAccuracy vs Gamma, pearson coefficient = '+str((round(pearsonR[0],4),round(pearsonR[1],4)))
    title=trial+'\nAccuracy vs Gamma, pearson coefficient = '+(str((round(pearsonR[0],4),round(pearsonR[1],4))) if round(pearsonR[1],4)!=0 else '('+str(round(pearsonR[0],4))+', <0.0001)')
    plt.gcf().suptitle(title,y=0.995)
    
  #  df_subset['invert']=1-df_subset[resultparam]
  #  df_subset.plot(x=Gammaparam,y='invert',kind='scatter',loglog=True,ylim=(0,1),
  #                      title='Accuracy vs Gamma, pearson coefficient = '+str((round(pearsonR[0],4),round(pearsonR[1],4))))

def GNB_smoothing_significance(df,modelparam,resultparam,Smoothparam,trial=None):
    df_subset=df[[modelparam,resultparam,Smoothparam]]
    df_subset[modelparam]=[x[0] for x in df_subset[modelparam]]
    df_subset=df_subset.loc[df_subset[modelparam]==4].reset_index(drop=True)
    df_subset[Smoothparam]=[x[0] for x in df_subset[Smoothparam]]
    
    pearsonR=stats.pearsonr(df_subset[Smoothparam],df_subset[resultparam])
    
    if round(pearsonR[1],4)==0:
        title=trial+'\nAccuracy vs Smoothing, pearson coefficient = '+'('+str(round(pearsonR[0],4))+', <0.0001)'
    else:
        title=trial+'\nAccuracy vs Smoothing, pearson coefficient = '+str((round(pearsonR[0],4),round(pearsonR[1],4)))
    df_subset.plot(x=Smoothparam,y=resultparam,kind='scatter',logx=True,#ylim=(0.8,1),
                        )#title=title,titleheight=0.995)
    plt.gcf().suptitle(title,y=0.995)

def kNN_k_significance(df,modelparam,resultparam,Kparam,trial=None):
    df_subset=df[[modelparam,resultparam,Kparam]]
    df_subset[modelparam]=[x[0] for x in df_subset[modelparam]]
    df_subset=df_subset.loc[df_subset[modelparam]==1].reset_index(drop=True)
    df_subset[Kparam]=[x[0] for x in df_subset[Kparam]]
    
    spearmanR=stats.spearmanr(df_subset[Kparam],df_subset[resultparam])
    
    title=trial+'\nAccuracy vs k, spearman rho = '+(str((round(spearmanR[0],4),round(spearmanR[1],4))) if round(spearmanR[1],4)!=0 else '('+str(round(spearmanR[0],4))+', <0.0001)')
    df_subset.plot(x=Kparam,y=resultparam,kind='scatter')
    plt.gcf().suptitle(title,y=0.995)
    
def QDA_reg_significance(df,modelparam,resultparam,Regparam,trial=None):
    df_subset=df[[modelparam,resultparam,Regparam]]
    df_subset[modelparam]=[x[0] for x in df_subset[modelparam]]
    df_subset=df_subset.loc[df_subset[modelparam]==3].reset_index(drop=True)
    df_subset[Regparam]=[x[0] for x in df_subset[Regparam]]
    
    pearsonR=stats.pearsonr(df_subset[Regparam],df_subset[resultparam])
    
    title=trial+'\nAccuracy vs Regularisation, pearson coefficient = '+(str((round(pearsonR[0],4),round(pearsonR[1],4))) if round(pearsonR[1],4)!=0 else '('+str(round(pearsonR[0],4))+', <0.0001)')
    df_subset.plot(x=Regparam,y=resultparam,kind='scatter')
    plt.gcf().suptitle(title,y=0.995)
    
def RF_trees_significance(df,modelparam,resultparam,Treesparam,trial=None):
    df_subset=df[[modelparam,resultparam,Treesparam]]
    df_subset[modelparam]=[x[0] for x in df_subset[modelparam]]
    df_subset=df_subset.loc[df_subset[modelparam]==0].reset_index(drop=True)
    df_subset[Treesparam]=[x[0] for x in df_subset[Treesparam]]
    
    spearmanR=stats.spearmanr(df_subset[Treesparam],df_subset[resultparam])
    
    title=trial+'\nAccuracy vs # of trees, spearman rho = '+(str((round(spearmanR[0],4),round(spearmanR[1],4))) if round(spearmanR[1],4)!=0 else '('+str(round(spearmanR[0],4))+', <0.0001)')
    df_subset.plot(x=Treesparam,y=resultparam,kind='scatter')
    plt.gcf().suptitle(title,y=0.995)

if __name__=='__main__':
    test_LDAs=False
    test_decisions=False
    test_hierarch = False
    plt.rcParams['figure.dpi']=150
    testGNB=False
    
    test_featfuse_SVMs=False
    test_eeg_SVMs=False
    test_knns=True
    test_QDAs=True
    test_RFs=True
    
    pathBespokeEEG=r"/home/michael/Documents/Aston/MultimodalFW/rq1-unimodal-opt-forGNB/bespoke_just_eeg/trials_obj.p"
    _,_,eegBespoke=fuse.load_results_obj(pathBespokeEEG)

    pathBespokeEMG=r"/home/michael/Documents/Aston/MultimodalFW/rq1-unimodal-opt-forGNB/bespoke_just_emg/trials_obj.p"
    _,_,emgBespoke=fuse.load_results_obj(pathBespokeEMG)

    pathGenEEG=r"/home/michael/Documents/Aston/MultimodalFW/rq1-unimodal-opt-forGNB/Gen_EEG/trials_obj.p"
    _,_,eegGen=fuse.load_results_obj(pathGenEEG)

    pathGenEMG=r"/home/michael/Documents/Aston/MultimodalFW/rq1-unimodal-opt-forGNB/Gen_EMG/trials_obj.p"
    _,_,emgGen=fuse.load_results_obj(pathGenEMG)
    
    if test_knns:
        kNN_k_significance(eegBespoke,'eeg model','eeg_mean_acc','eeg.knn.k',trial='kNNs in Bespoke Unimodal EEG System Optimisation')
        kNN_k_significance(emgBespoke,'emg model','emg_mean_acc','emg.knn.k',trial='kNNs in Bespoke Unimodal EMG System Optimisation')
        kNN_k_significance(eegGen,'eeg model','eeg_mean_acc','eeg.knn.k',trial='kNNs in Generalist Unimodal EEG System Optimisation')
        kNN_k_significance(emgGen,'emg model','emg_mean_acc','emg.knn.k',trial='kNNs in Generalist Unimodal EMG System Optimisation')
    
    if test_QDAs:
       QDA_reg_significance(eegBespoke,'eeg model','eeg_mean_acc','eeg.qda.regularisation',trial='QDAs in Bespoke Unimodal EEG System Optimisation')
       QDA_reg_significance(emgBespoke,'emg model','emg_mean_acc','emg.qda.regularisation',trial='QDAs in Bespoke Unimodal EMG System Optimisation')
       QDA_reg_significance(eegGen,'eeg model','eeg_mean_acc','eeg.qda.regularisation',trial='QDAs in Generalist Unimodal EEG System Optimisation')
       QDA_reg_significance(emgGen,'emg model','emg_mean_acc','emg.qda.regularisation',trial='QDAs in Generalist Unimodal EMG System Optimisation')
       
    if test_RFs:
       RF_trees_significance(eegBespoke,'eeg model','eeg_mean_acc','eeg_ntrees',trial='RFs in Bespoke Unimodal EEG System Optimisation')
       RF_trees_significance(emgBespoke,'emg model','emg_mean_acc','emg.RF.ntrees',trial='RFs in Bespoke Unimodal EMG System Optimisation')
       RF_trees_significance(eegGen,'eeg model','eeg_mean_acc','eeg_ntrees',trial='RFs in Generalist Unimodal EEG System Optimisation')
       RF_trees_significance(emgGen,'emg model','emg_mean_acc','emg.RF.ntrees',trial='RFs in Generalist Unimodal EMG System Optimisation')
    
    raise
    if test_featfuse_SVMs:
        pathBespokeFeatSep=r"/home/michael/Documents/Aston/MultimodalFW/rq1-featfuse-opt-res/featlevel (sep)/trials_obj.p"
        _,_,bespokeFeatSep=fuse.load_results_obj(pathBespokeFeatSep)
        print('\n Bespoke Feature-Level (separate selection):')
        SVM_params_significance(bespokeFeatSep,'featfuse model','fusion_mean_acc','featfuse.svm.c','featfuse.svm.gamma',trial='SVMs in Bespoke Feature-Fusion (Separate Sel) Optimisation')
        
        pathBespokeFeatJoin=r"/home/michael/Documents/Aston/MultimodalFW/rq1-featfuse-opt-res/featlevel_joint/trials_obj.p"
        _,_,bespokeFeatJoin=fuse.load_results_obj(pathBespokeFeatJoin)
        print('\n Bespoke Feature-Level (joint selection):')
        SVM_params_significance(bespokeFeatJoin,'featfuse model','fusion_mean_acc','featfuse.svm.c','featfuse.svm.gamma',trial='SVMs in Bespoke Feature-Fusion (Joint Sel) Optimisation')
        
        raise
    if test_eeg_SVMs:
        pathBespokeEEG=r"/home/michael/Documents/Aston/MultimodalFW/rq1-unimodal-opt-forGNB/bespoke_just_eeg/trials_obj.p"
        _,_,bespokeEEG=fuse.load_results_obj(pathBespokeEEG)
        SVM_params_significance(bespokeEEG,'eeg model','eeg_mean_acc','eeg.svm.c','eeg.svm.gamma',trial='SVMs in Bespoke Unimodal EEG System Optimisation')
    raise    
    if testGNB:
        pathBespokeEEG=r"/home/michael/Documents/Aston/MultimodalFW/rq1-unimodal-opt-forGNB/bespoke_just_eeg/trials_obj.p"
        _,_,eegBespoke=fuse.load_results_obj(pathBespokeEEG)
        print('\n Bespoke EEG-Only:')
        GNB_smoothing_significance(eegBespoke,'eeg model','eeg_mean_acc','eeg.gnb.smoothing',trial='GNBs in Bespoke Unimodal EEG System Optimisation')
        
        pathBespokeEMG=r"/home/michael/Documents/Aston/MultimodalFW/rq1-unimodal-opt-forGNB/bespoke_just_emg/trials_obj.p"
        _,_,emgBespoke=fuse.load_results_obj(pathBespokeEMG)
        print('\n Bespoke EMG-Only:')
        GNB_smoothing_significance(emgBespoke,'emg model','emg_mean_acc','emg.gnb.smoothing',trial='GNBs in Bespoke Unimodal EMG System Optimisation')
        
        pathGenEEG=r"/home/michael/Documents/Aston/MultimodalFW/rq1-unimodal-opt-forGNB/Gen_EEG/trials_obj.p"
        _,_,eegGen=fuse.load_results_obj(pathGenEEG)
        print('\n Gen EEG-Only:')
        GNB_smoothing_significance(eegGen,'eeg model','eeg_mean_acc','eeg.gnb.smoothing',trial='GNBs in Generalist Unimodal EEG System Optimisation')
        
        pathGenEMG=r"/home/michael/Documents/Aston/MultimodalFW/rq1-unimodal-opt-forGNB/Gen_EMG/trials_obj.p"
        _,_,emgGen=fuse.load_results_obj(pathGenEMG)
        print('\n Gen EMG-Only:')
        GNB_smoothing_significance(emgGen,'emg model','emg_mean_acc','emg.gnb.smoothing',trial='GNBs in Generalist Unimodal EMG System Optimisation')
    raise
    if test_hierarch:
        models=['RF','KNN','LDA','QDA','GNB','SVM']
        model_dict=dict(zip(range(0,len(models)+1),models))
        
        pathBespokeHierarch='/home/michael/Documents/Aston/MultimodalFW/rq1-hierarch-opt-results/hierarchical/trials_obj.p'
        _,_,BespHierarch=fuse.load_results_obj(pathBespokeHierarch)
        figBespHier,statsBespHier=hierarch_topmodel(BespHierarch,'fusion_mean_acc','emg model',title='Mean accuracy per EMG+ model in\n Bespoke Hierarchical fusion optimisation',titleheight=0.995)
        figBespHierLow=hierarch_lowmodel(BespHierarch,'fusion_mean_acc','eeg model',title='Mean accuracy per EEG component model in\n Bespoke Hierarchical fusion optimisation',titleheight=0.995)
        
        pathBespokeInvHierarch='/home/michael/Documents/Aston/MultimodalFW/rq1-hierarch-opt-results/hierarchical_inv/trials_obj.p'
        _,_,BespInvHierarch=fuse.load_results_obj(pathBespokeInvHierarch)
        figBespInvHier,statsBespInvHier=hierarch_topmodel(BespInvHierarch,'fusion_mean_acc','eeg model',title='Mean accuracy per EEG+ model in\n Bespoke Inverse Hierarchical fusion optimisation',titleheight=0.995)
        figBespInvHierLow=hierarch_lowmodel(BespInvHierarch,'fusion_mean_acc','emg model',title='Mean accuracy per EMG component model in\n Bespoke Inverse Hierarchical fusion optimisation',titleheight=0.995)
        
      #  figBespHier.savefig('/home/michael/Documents/Aston/MultimodalFW/rq1-hierarch-opt-results/hierarch-bespoke-topmodel-box.png')
      #  figBespInvHier.savefig('/home/michael/Documents/Aston/MultimodalFW/rq1-hierarch-opt-results/invhierarch-bespoke-topmodel-box.png')
        figBespHierLow.savefig('/home/michael/Documents/Aston/MultimodalFW/rq1-hierarch-opt-results/hierarch-bespoke-lowmodel-box.png')
        figBespInvHierLow.savefig('/home/michael/Documents/Aston/MultimodalFW/rq1-hierarch-opt-results/invhierarch-bespoke-lowmodel-box.png')
        
        models=['RF','KNN','LDA','QDA','GNB']
        model_dict=dict(zip(range(0,len(models)+1),models))
        
        pathGenHierarch='/home/michael/Documents/Aston/MultimodalFW/rq1-hierarch-opt-results/Gen_hierarch_prob/trials_obj.p'
        _,_,GenHierarch=fuse.load_results_obj(pathGenHierarch)
        figGenHier,statsGenHier=hierarch_topmodel(GenHierarch,'fusion_mean_acc','emg model',title='Mean accuracy per EMG+ model in\n Generalist Hierarchical fusion optimisation',titleheight=0.995)
        figGenHierLow=hierarch_lowmodel(GenHierarch,'fusion_mean_acc','eeg model',title='Mean accuracy per EEG component model in\n Generalist Hierarchical fusion optimisation',titleheight=0.995)
        
        pathGenInvHierarch='/home/michael/Documents/Aston/MultimodalFW/rq1-hierarch-opt-results/Gen_inv_hierarch_prob/trials_obj.p'
        _,_,GenInvHierarch=fuse.load_results_obj(pathGenInvHierarch)
        figGenInvHier,statsGenInvHier=hierarch_topmodel(GenInvHierarch,'fusion_mean_acc','eeg model',title='Mean accuracy per EEG+ model in\n Generalist Inverse Hierarchical fusion optimisation',titleheight=0.995)
        figGenInvHierLow=hierarch_lowmodel(GenInvHierarch,'fusion_mean_acc','emg model',title='Mean accuracy per EMG component model in\n Generalist Inverse Hierarchical fusion optimisation',titleheight=0.995)
        
      #  figGenHier.savefig('/home/michael/Documents/Aston/MultimodalFW/rq1-hierarch-opt-results/hierarch-gen-topmodel-box.png')
      #  figGenInvHier.savefig('/home/michael/Documents/Aston/MultimodalFW/rq1-hierarch-opt-results/invhierarch-gen-topmodel-box.png')
        figGenHierLow.savefig('/home/michael/Documents/Aston/MultimodalFW/rq1-hierarch-opt-results/hierarch-gen-lowmodel-box.png')
        figGenInvHierLow.savefig('/home/michael/Documents/Aston/MultimodalFW/rq1-hierarch-opt-results/invhierarch-gen-lowmodel-box.png')
        
    raise
    if test_decisions:
        algs=['mean','3_1_emg','3_1_eeg','opt_weight','highest_conf','svm','lda','rf']
        alg_dict=dict(zip(range(0,len(algs)+1),algs))
        models=['RF','KNN','LDA','QDA','GNB','SVM']
        model_dict=dict(zip(range(0,len(models)+1),models))
        
        pathBespokeDec=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Bespoke_20DevSet\decision\trials_obj.p"
        _,_,decBespoke=fuse.load_results_obj(pathBespokeDec)
        print('\n Bespoke Decision-Level:')
        fig=fusalg_significance(decBespoke,'fusion_mean_acc','fusion algorithm','highest_conf',title='Mean accuracy per fusion alg in Bespoke Decision fusion optimisation')
        fig.savefig(r"C:\Users\pritcham\Documents\RQ1_plots_stats\bespoke_DecAlgs_box.png")
        fig=model_significance(decBespoke, 'fusion_mean_acc', 'emg model', 'SVM',title='Mean accuracy per EMG classifier in Bespoke Decision fusion optimisation')
        fig.savefig(r"C:\Users\pritcham\Documents\RQ1_plots_stats\bespoke_Decision_EMG_box.png")
        fig=model_significance(decBespoke, 'fusion_mean_acc', 'eeg model', 'RF',title='Mean accuracy per EEG classifier in Bespoke Decision fusion optimisation')
        fig.savefig(r"C:\Users\pritcham\Documents\RQ1_plots_stats\bespoke_Decision_EEG_box.png")
    
        
        models=['RF','KNN','LDA','QDA','GNB']
        model_dict=dict(zip(range(0,len(models)+1),models))
        pathGenDec=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Generalist_20DevSet\Gen_Decision\trials_obj.p"
        _,_,decGeneralist=fuse.load_results_obj(pathGenDec)
        print('\n Generalist Decision-Level:')
        fig=fusalg_significance(decGeneralist,'fusion_mean_acc','fusion algorithm','highest_conf',title='Mean accuracy per fusion alg in Generalist Decision fusion optimisation')
        fig.savefig(r"C:\Users\pritcham\Documents\RQ1_plots_stats\gen_DecAlgs_box.png")
        fig=model_significance(decGeneralist, 'fusion_mean_acc', 'emg model', 'LDA',title='Mean accuracy per EMG classifier in Generalist Decision fusion optimisation')
        fig.savefig(r"C:\Users\pritcham\Documents\RQ1_plots_stats\gen_Decision_EMG_box.png")
        fig=model_significance(decGeneralist, 'fusion_mean_acc', 'eeg model', 'LDA',title='Mean accuracy per EEG classifier in Generalist Decision fusion optimisation')
        fig.savefig(r"C:\Users\pritcham\Documents\RQ1_plots_stats\gen_Decision_EEG_box.png")
    
    if test_LDAs:
        models=['RF','KNN','LDA','QDA','GNB','SVM']
        model_dict=dict(zip(range(0,len(models)+1),models))
        
        
        
        pathBespokeEEG=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Bespoke_20DevSet\just_eeg\trials_obj.p"
        _,_,eegBespoke=fuse.load_results_obj(pathBespokeEEG)
        print('\n Bespoke EEG-Only:')
        fig=model_significance(eegBespoke,'eeg_mean_acc','eeg model','LDA',title='Mean accuracy per classifier in Bespoke EEG-Only optimisation')
        fig.savefig(r"C:\Users\pritcham\Documents\RQ1_plots_stats\bespoke_eegOnly_box.png")
        LDA_solver_significance(eegBespoke,'eeg model','eeg_mean_acc','eeg.LDA_solver','eeg.lda.shrinkage')
        
        
        
        pathBespokeEMG=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Bespoke_20DevSet\just_emg\trials_obj.p"
        _,_,emgBespoke=fuse.load_results_obj(pathBespokeEMG)
        print('\n Bespoke EMG-Only:')
        fig=model_significance(emgBespoke,'emg_mean_acc','emg model','LDA',title='Mean accuracy per classifier in Bespoke EMG-Only optimisation')
        LDA_solver_significance(emgBespoke,'emg model','emg_mean_acc','emg.LDA_solver','emg.lda.shrinkage')
        
        
        #pathBespokeEMG=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Bespoke_20DevSet\just_emg\trials_obj.p"
        #_,_,emgBespoke=fuse.load_results_obj(pathBespokeEMG)
        models=['RF','KNN','LDA','QDA','GNB','SVM']
        model_dict=dict(zip(range(0,len(models)+1),models))
        print('\n Bespoke EMG-Only:')
        fig=model_significance(emgBespoke,'emg_mean_acc','emg model','SVM',title='Mean accuracy per classifier in Bespoke EMG-Only optimisation')
        SVM_params_significance(emgBespoke,'emg model','emg_mean_acc','emg.svm.c','emg.svm.gamma')
        
        
        
        pathBespokeFeatSep=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Bespoke_20DevSet\featlevel\trials_obj.p"
        _,_,bespokeFeatSep=fuse.load_results_obj(pathBespokeFeatSep)
        print('\n Bespoke Feature-Level (separate selection):')
        fig=model_significance(bespokeFeatSep,'fusion_mean_acc','featfuse model','LDA',title='Mean accuracy per classifier in Bespoke Feature-level (separate) optimisation')
        fig.savefig(r"C:\Users\pritcham\Documents\RQ1_plots_stats\bespoke_featsep_box.png")
        LDA_solver_significance(bespokeFeatSep,'featfuse model','fusion_mean_acc','featfuse.LDA_solver','featfuse.lda.shrinkage')
        
        
        pathBespokeFeatJoin=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Bespoke_20DevSet\featlevel_joint\trials_obj.p"
        _,_,bespokeFeatJoin=fuse.load_results_obj(pathBespokeFeatJoin)
        print('\n Bespoke Feature-Level (joint selection):')
        fig=model_significance(bespokeFeatJoin,'fusion_mean_acc','featfuse model','LDA',title='Mean accuracy per classifier in Bespoke Feature-level (joint) optimisation')
        fig.savefig(r"C:\Users\pritcham\Documents\RQ1_plots_stats\bespoke_featjoin_box.png")
        LDA_solver_significance(bespokeFeatJoin,'featfuse model','fusion_mean_acc','featfuse.LDA_solver','featfuse.lda.shrinkage')
    
    
        '''_____GENERALIST_____'''
    
        models=['RF','KNN','LDA','QDA','GNB']
        model_dict=dict(zip(range(0,len(models)+1),models))
        
        pathGenEEG=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Generalist_20DevSet\Gen_EEG\trials_obj.p"
        _,_,eegGen=fuse.load_results_obj(pathGenEEG)
        print('\n Generalist EEG-Only:')
        fig=model_significance(eegGen,'eeg_mean_acc','eeg model','LDA',title='Mean accuracy per classifier in Generalist EEG-Only optimisation')
        fig.savefig(r"C:\Users\pritcham\Documents\RQ1_plots_stats\gen_eegOnly_box.png")
        LDA_solver_significance(eegGen,'eeg model','eeg_mean_acc','eeg.LDA_solver','eeg.lda.shrinkage')
        
        
        pathGenEMG=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Generalist_20DevSet\Gen_EMG\trials_obj.p"
        _,_,emgGen=fuse.load_results_obj(pathGenEMG)
        print('\n Generalist EMG-Only:')
        fig=model_significance(emgGen,'emg_mean_acc','emg model','LDA',title='Mean accuracy per classifier in Generalist EMG-Only optimisation')
        fig.savefig(r"C:\Users\pritcham\Documents\RQ1_plots_stats\gen_emgOnly_box.png")
        LDA_solver_significance(emgGen,'emg model','emg_mean_acc','emg.LDA_solver','emg.lda.shrinkage')
        
        
        pathGenFeatSep=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Generalist_20DevSet\Gen_feat_sep\trials_obj.p"
        _,_,genFeatSep=fuse.load_results_obj(pathGenFeatSep)
        print('\n Generalist Feature-Level (separate selection):')
        fig=model_significance(genFeatSep,'fusion_mean_acc','featfuse model','LDA',title='Mean accuracy per classifier in Generalist Feature-level (separate) optimisation')
        fig.savefig(r"C:\Users\pritcham\Documents\RQ1_plots_stats\gen_featsep_box.png")
        LDA_solver_significance(genFeatSep,'featfuse model','fusion_mean_acc','featfuse.LDA_solver','featfuse.lda.shrinkage')
        
        
        pathGenFeatJoin=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Generalist_20DevSet\Gen_feat_joint\trials_obj.p"
        _,_,genFeatJoin=fuse.load_results_obj(pathGenFeatJoin)
        print('\n Generalist Feature-Level (joint selection):')
        fig=model_significance(genFeatJoin,'fusion_mean_acc','featfuse model','LDA',title='Mean accuracy per classifier in Generalist Feature-level (joint) optimisation')
        fig.savefig(r"C:\Users\pritcham\Documents\RQ1_plots_stats\gen_featjoin_box.png")
        LDA_solver_significance(genFeatJoin,'featfuse model','fusion_mean_acc','featfuse.LDA_solver','featfuse.lda.shrinkage')
    


    raise
    
    bespoke_models=['RF','KNN','LDA','QDA','GNB','SVM']
    model_dict=dict(zip(range(0,len(bespoke_models)+1),bespoke_models))
    
    pathBespokeEEG=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Bespoke_20DevSet\just_eeg\trials_obj.p"
    _,_,eegBespoke=fuse.load_results_obj(pathBespokeEEG)
    per_eegmodel=boxplot_param(eegBespoke,'eeg model','eeg_mean_acc',xlabs=bespoke_models,
                               title='Mean accuracy per classifier in Bespoke EEG-Only optimisation')
    per_eegmodel.savefig(r"C:\Users\pritcham\Documents\RQ1_plots_stats\eegOnly_box.png")
    
    eeg_stat_test=eegBespoke[['eeg model','fusion_mean_acc']]
    eeg_stat_test['eeg model']=[x[0] for x in eeg_stat_test['eeg model']]
    eeg_stat_test['eeg model']=[model_dict[x] for x in eeg_stat_test['eeg model']]
    eeg_groups = [eeg_stat_test[eeg_stat_test['eeg model'] == group]['fusion_mean_acc'] for group in bespoke_models]
    #sorting the above by bespoke_models instead of #stat_test['eeg model'].unique()] #so that order is preserved
    fstat, pvalue = stats.f_oneway(*eeg_groups)
    print('anova on all ',fstat, pvalue)
    #tukey=stats.tukey_hsd(*groups)
    #print(tukey)
    winner='LDA'
    win_idx=next(key for key,value in model_dict.items() if value==winner)
    for i in range(len(eeg_groups)):
        if i==win_idx:
            continue
        if np.std(eeg_groups[win_idx]) - np.std(eeg_groups[i]) > 0.05:
            #use math.isclose() ?
            tscore,pval=stats.ttest_ind(eeg_groups[win_idx],eeg_groups[i],equal_var=False)
            print('Welch\'s t on ',model_dict[win_idx],' & ',model_dict[i],': ',tscore,pval)
        else:
           # tscore,pval=stats.ttest_ind(groups[win_idx],groups[i],equal_var=True)
           # print('Student\'s t on ',model_dict(win_idx),' & ',model_dict(i),': ',tscore,pval)
            '''recommendations that variance equality not be checked for
            https://bpspsychub.onlinelibrary.wiley.com/doi/abs/10.1348/000711004849222
            and Welchs used throughout instead, despite being worse at low sample size'''
            tscore,pval=stats.ttest_ind(eeg_groups[win_idx],eeg_groups[i],equal_var=False)
            print('Welch\'s t on ',model_dict[win_idx],' & ',model_dict[i],': ',tscore,pval)
     
    eegBespokeLDAs=eegBespoke[['eeg model','eeg_mean_acc','eeg.LDA_solver','eeg.lda.shrinkage']]
    eegBespokeLDAs['eeg model']=[x[0] for x in eegBespokeLDAs['eeg model']]
    eegBespokeLDAs=eegBespokeLDAs.loc[eegBespokeLDAs['eeg model']==2].reset_index(drop=True)
    eegBespokeLDAs['eeg.LDA_solver']=[x[0] for x in eegBespokeLDAs['eeg.LDA_solver']]
    eegBespokeLDAs['eeg.lda.shrinkage']=[x[0] for x in eegBespokeLDAs['eeg.lda.shrinkage']]
    
    solvers=['svd','lsqr','eigen']
    solver_dict=dict(zip(range(0,len(solvers)+1),solvers))
    
    solver_groups = [eegBespokeLDAs[eegBespokeLDAs['eeg.LDA_solver'] == group]['eeg_mean_acc'] for group in eegBespokeLDAs['eeg.LDA_solver'].unique()]
    fstat, pvalue = stats.f_oneway(*solver_groups)
    print('anova on all ',fstat, pvalue)
    per_LDAsolver=boxplot_param(eegBespokeLDAs,'eeg.LDA_solver','eeg_mean_acc',xlabs=solvers,
                               title='Mean accuracy per LDA solver, ANOVA: f='+str(round(fstat,4))+', p='+str(round(pvalue,4)))
    
    print('below only looks for linear correlation, optimiser looks for local minimum')
    LDAs_perSolver=[eegBespokeLDAs[eegBespokeLDAs['eeg.LDA_solver']==group].reset_index(drop=True) for group in eegBespokeLDAs['eeg.LDA_solver'].unique()]
    eegBespokeLDAs_noSVD=eegBespokeLDAs.loc[eegBespokeLDAs['eeg.LDA_solver']!=0].reset_index(drop=True)
    pearsonR=stats.pearsonr(eegBespokeLDAs_noSVD['eeg.lda.shrinkage'],eegBespokeLDAs_noSVD['eeg_mean_acc'])
    eegBespokeLDAs_noSVD.plot(x='eeg.lda.shrinkage',y='eeg_mean_acc',kind='scatter',
                        title='Accuracy vs LDA shrinkage, pearson coefficient = '+str((round(pearsonR[0],4),round(pearsonR[1],4))))
    
    pearsonR=stats.pearsonr(LDAs_perSolver[1]['eeg.lda.shrinkage'],LDAs_perSolver[1]['eeg_mean_acc'])
    LDAs_perSolver[1].plot(x='eeg.lda.shrinkage',y='eeg_mean_acc',kind='scatter',
                        title='Accuracy vs shrinkage, solver: '+str(solver_dict[1])+', pearson coefficient = '+str((round(pearsonR[0],4),round(pearsonR[1],4))))
    
    pearsonR=stats.pearsonr(LDAs_perSolver[2]['eeg.lda.shrinkage'],LDAs_perSolver[2]['eeg_mean_acc'])
    LDAs_perSolver[2].plot(x='eeg.lda.shrinkage',y='eeg_mean_acc',kind='scatter',
                        title='Accuracy vs shrinkage, solver: '+str(solver_dict[2])+', pearson coefficient = '+str((round(pearsonR[0],4),round(pearsonR[1],4))))
    
    
        
    #t_ind, p_ind = stats.ttest_ind(groups[2],groups[3])
    #print('independent t on LDA & QDA ',t_ind, p_ind)
            
            
    pathBespokeEMG=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\results\Bespoke_20DevSet\just_emg\trials_obj.p"
    _,_,emgBespoke=fuse.load_results_obj(pathBespokeEMG)
    per_emgmodel=boxplot_param(emgBespoke,'emg model','emg_mean_acc',xlabs=bespoke_models,
                               title='Mean accuracy per emg model type in emg-Only optimisation')
    
    stat_test=emgBespoke[['emg model','fusion_mean_acc']]
    stat_test['emg model']=[x[0] for x in stat_test['emg model']]
    stat_test['emg model']=[model_dict[x] for x in stat_test['emg model']]
    groups = [stat_test[stat_test['emg model'] == group]['fusion_mean_acc'] for group in bespoke_models]
    #sorting the above by bespoke_models instead of #stat_test['emg model'].unique()] #so that order is preserved
    fstat, pvalue = stats.f_oneway(*groups)
    print('anova on all ',fstat, pvalue)
    #tukey=stats.tukey_hsd(*groups)
    #print(tukey)
    winner='SVM'
    win_idx=next(key for key,value in model_dict.items() if value==winner)
    for i in range(len(groups)):
        if i==win_idx:
            continue
        if np.std(groups[win_idx]) - np.std(groups[i]) > 0.05:
            tscore,pval=stats.ttest_ind(groups[win_idx],groups[i],equal_var=False)
            print('Welch\'s t on ',model_dict[win_idx],' & ',model_dict[i],': ',tscore,pval)
        else:
           # tscore,pval=stats.ttest_ind(groups[win_idx],groups[i],equal_var=True)
           # print('Student\'s t on ',model_dict(win_idx),' & ',model_dict(i),': ',tscore,pval)
            '''recommendations that variance equality not be checked for
            https://bpspsychub.onlinelibrary.wiley.com/doi/abs/10.1348/000711004849222
            and Welchs used throughout instead, despite being worse at low sample size'''
            tscore,pval=stats.ttest_ind(groups[win_idx],groups[i],equal_var=False)
            print('Welch\'s t on ',model_dict[win_idx],' & ',model_dict[i],': ',tscore,pval)
    

