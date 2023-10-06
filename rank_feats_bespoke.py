# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 18:56:25 2023

@author: pritcham
"""


import testFusion as fuse
import handleML as ml
from sklearn.manifold import TSNE
import params as params
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import handleFeats as feats
import random as random
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import seaborn as sns



def get_session_masks(featset):
    masks=[featset['ID_run']== run for run in np.sort(featset['ID_run'].unique())] 
    return masks

def rank_feats_l1_df(data,sparsityC):
    target=data['Label']
    attribs=data.drop(columns=['Label'])   
    lsvc = LinearSVC(C=sparsityC, penalty="l1", dual=False).fit(attribs, target)
    #getting all nonzero feats
    #model = SelectFromModel(lsvc, prefit=True)
    #getting feats up to maxfeats, with no threshold of rating so as to ensure never <maxfeats
    coefs_SVC=lsvc.coef_
    importances = np.linalg.norm(coefs_SVC, axis=0, ord=1)
    #col_names=model.get_feature_names_out(data.attribs.columns.values)
    return coefs_SVC, importances


broad_cats={'sum_delta':'Frequency',
           'sum_theta':'Frequency',
           'sum_gamma':'Frequency',
           'sum_alpha':'Frequency',
           'sum_beta':'Frequency',
           'mean':'Distribution',
           'mean_q1':'Distribution',
           'mean_q2':'Distribution',
           'mean_q3':'Distribution',
           'mean_q4':'Distribution',
           'mean_d_q1q4':'\u0394 Distribution',
           'mean_d_q1q3':'\u0394 Distribution',
           'mean_d_q1q2':'\u0394 Distribution',
           'mean_d_q2q4':'\u0394 Distribution',
           'mean_d_q2q3':'\u0394 Distribution',
           'mean_d_q3q4':'\u0394 Distribution',
           'mean_d_h2h1':'\u0394 Distribution',
           'std':'Distribution',
           'std_d_h2h1':'\u0394 Distribution',
           'skew':'Distribution',
           'kurt':'Distribution',
           'covM':'Covariance',
           'logcovM':'Covariance',
           'eigenval':'Covariance',
           'max':'Extrema',
           'min':'Extrema',
           'min_q1':'Extrema',
           'min_q2':'Extrema',
           'min_q3':'Extrema',
           'min_q4':'Extrema',
           'max_q1':'Extrema',
           'max_q2':'Extrema',
           'max_q3':'Extrema',
           'max_q4':'Extrema',
           'min_d_q1q2':'\u0394 Extrema',
           'min_d_q1q3':'\u0394 Extrema',
           'min_d_q1q4':'\u0394 Extrema',
           'min_d_q2q3':'\u0394 Extrema',
           'min_d_q2q4':'\u0394 Extrema',
           'min_d_q3q4':'\u0394 Extrema',
           'max_d_q1q2':'\u0394 Extrema',
           'max_d_q1q3':'\u0394 Extrema',
           'max_d_q1q4':'\u0394 Extrema',
           'max_d_q2q3':'\u0394 Extrema',
           'max_d_q2q4':'\u0394 Extrema',
           'max_d_q3q4':'\u0394 Extrema',
           'min_d_h2h1':'\u0394 Extrema',
           'max_d_h2h1':'\u0394 Extrema',
           }


emg_set_path=params.jeong_emg_noholdout,
eeg_set_path=params.jeong_eeg_noholdout,
#eeg_set_path=params.eeg_jeongSyncCSP_feats,
#eeg_set_path=params.eeg_jeong_feats,
            
emg_set=ml.pd.read_csv(emg_set_path[0],delimiter=',')
eeg_set=ml.pd.read_csv(eeg_set_path[0],delimiter=',')
#emg_set,eeg_set=fuse.balance_set(emg_set,eeg_set)


per_ppt = True
plotemg=False
ploteeg=True
n_components = 2
if per_ppt:
    emg_masks=fuse.get_ppt_split_flexi(emg_set)
    eeg_masks=fuse.get_ppt_split_flexi(eeg_set)
    col_rankings_eeg=[]
    col_selections_eeg=[]
    for idx,emg_mask in enumerate(emg_masks):
        eeg_mask=eeg_masks[idx]
        emg_ppt = emg_set[emg_mask]
        eeg_ppt = eeg_set[eeg_mask]


        split_sessions=False
        
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
        gest_strat=pd.DataFrame([gest_perfs,[perf.split('.')[1][-1] for perf in gest_perfs]]).transpose()
        train_split,test_split=fuse.train_test_split(gest_strat,test_size=0.33,random_state=random_split,stratify=gest_strat[1])

        eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(train_split[0])]
        eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split[0])]
        emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(train_split[0])]
        emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]
        
        
        if plotemg:
            '''
            y_emg=emg_ppt.pop('Label')
            X_emg=emg_ppt
            X_emg=ml.drop_ID_cols(X_emg)
            X_emg,emgscaler=feats.scale_feats_train(X_emg,'standardise')
            
            X_emg['Label']=y_emg
            sel_cols_emg=feats.sel_percent_feats_df(X_emg,percent=15)
            X_emg.pop('Label')
            X_emg=X_emg.iloc[:,sel_cols_emg]
            
            tsne_emg = TSNE(n_components)
            
            print('starting emg tsne')
            tsne_result_emg = tsne_emg.fit_transform(X_emg)
            print(tsne_result_emg.shape)            
            plot_tsne(tsne_result_emg,y_emg,('emg ppt '+str(idx)))
            '''

        if ploteeg:
            
            if not split_sessions:
                y_eeg=eeg_train.pop('Label')
                X_eeg=eeg_train
                X_eeg=ml.drop_ID_cols(X_eeg)
                X_eeg,eegscaler=feats.scale_feats_train(X_eeg,'standardise')
            
                X_eeg['Label']=y_eeg
                sel_cols_eeg=feats.sel_feats_l1_df(X_eeg,sparsityC=0.005,maxfeats=40)
                _,importances=rank_feats_l1_df(X_eeg,sparsityC=0.005)
                X_eeg.pop('Label')
                X_eeg=X_eeg.iloc[:,sel_cols_eeg]
                chosencols=X_eeg.columns.values
                col_rankings_eeg.append(importances)
                col_selections_eeg.append(chosencols)
    col_rankings_eeg=pd.DataFrame(col_rankings_eeg,columns=(ml.drop_ID_cols(eeg_train)).columns.values)
    ppt0_rankings_eeg=col_rankings_eeg.iloc[0]
    ppt0_rankings_eeg=ppt0_rankings_eeg.rename_axis('feat').reset_index()
    ppt0_rankings_eeg['category']=ppt0_rankings_eeg['feat'].copy()
    ppt0_rankings_eeg['category']=ppt0_rankings_eeg['category'].apply(lambda x: x[:min([i for i, c in enumerate(x) if c=='_' and not any(char.isalpha() for char in x[i+1:])])])
    ppt0_rankings_eeg['broad_category']=[broad_cats[x.replace('lag1_','')] for x in ppt0_rankings_eeg['category']]
    ppt0_rankings_eeg=ppt0_rankings_eeg.rename(columns={0:'importance'})
    
    col_importances_eeg=col_rankings_eeg.T
    col_importances_eeg=col_importances_eeg.rename(columns={0:'importance_0',1:'importance_1',2:'importance_2',3:'importance_3',4:'importance_4',
                                                            5:'importance_5',6:'importance_6',7:'importance_7',8:'importance_8',9:'importance_9',
                                                            10:'importance_10',11:'importance_11',12:'importance_12',13:'importance_13',14:'importance_14',
                                                            15:'importance_15',16:'importance_16',17:'importance_17',18:'importance_18',19:'importance_19',
                                                            })
    col_importances_eeg=col_importances_eeg.rename_axis('feat').reset_index()
    col_importances_eeg['category']=col_importances_eeg['feat'].copy()
    col_importances_eeg['category']=col_importances_eeg['category'].apply(lambda x: x[:min([i for i, c in enumerate(x) if c=='_' and not any(char.isalpha() for char in x[i+1:])])])
    col_importances_eeg['broad_category']=[broad_cats[x.replace('lag1_','')] for x in col_importances_eeg['category']]
    col_importances_eeg=col_importances_eeg.sort_values('broad_category')
    for i in range(20):
        #col_importances_eeg=col_importances_eeg.sort_values(('importance_'+str(i)))
        #col_importances_eeg=col_importances_eeg.rename_axis(('rank_'+str(i))).reset_index()
        col_importances_eeg[('rank_'+str(i))]=len(col_importances_eeg)-col_importances_eeg[('importance_')+str(i)].rank(method='first')
        plt.figure(i)
        sns.scatterplot(x=('rank_'+str(i)), y=('importance_'+str(i)),data=col_importances_eeg, s=10, hue='broad_category', ec=None,palette='bright').set(title=('ppt '+str(i)))
        #leg=plt.legend()
        #for i,lab in leg.get_texts():

#    sns.scatterplot(x='rank', y='occurence', data=col_importances_eeg, hue='broad_category', ec=None,palette='muted')
    
    
    col_choices=pd.DataFrame(col_selections_eeg)
    #savepath=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\datasets\shared_feat_analysis\eeg-Bespoke-devsetOnly-feats.csv"
    #col_choices.to_csv(savepath,header=False,index=False)
    choices_unique=col_choices.T.copy().applymap(lambda x: x.replace('lag1_',''))
    choices_unique=choices_unique.apply(lambda x: x[~x.duplicated()]).fillna('')
    popular_unique=choices_unique.stack().value_counts().drop(index='')
    populars=popular_unique.rename_axis('feat').reset_index()
    populars['category']=populars['feat'].copy()
    populars['category']=populars['category'].apply(lambda x: x[:min([i for i, c in enumerate(x) if c=='_' and not any(char.isalpha() for char in x[i+1:])])])
    populars=populars.rename(columns={0:'occurence'})
    #populars=populars.rename_axis('rank').reset_index()
    #populars['rank']=len(populars)-populars['rank']
    
    
    populars['broad_category']=[broad_cats[x] for x in populars['category']]
    populars=populars.sort_values('broad_category')
    populars['rank']=len(populars)-populars['occurence'].rank(method='first')
    
    plt.figure()
    #populars=populars.sort_values('category')
    sns.scatterplot(x='rank', y='occurence', data=populars[populars['occurence']>3], hue='category', ec=None,palette='muted')
    print(populars['category'].value_counts())
    plt.figure()
    sns.scatterplot(x='rank', y='occurence', data=populars[populars['occurence']>1], hue='broad_category', ec=None,palette='muted')
    print(populars['broad_category'].value_counts())
    plt.figure()
    sns.scatterplot(x='rank', y='occurence', data=populars[populars['occurence']>6], hue='feat', ec=None,palette='Set2')
    
    plt.scatter(populars['rank'],populars['occurence'],c=pd.factorize(populars['category'])[0],cmap='Dark2',label=populars['category'].unique())
    plt.legend(pd.factorize(populars['category'])[0])
    
    
    fig, ax = plt.subplots()
    grouped = populars.groupby('category')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='rank', y='occurence', label=key, color=pd.factorize(populars['category'])[0])
    plt.show()
    #color_map = dict(zip(populars['category'].unique(), rgb_values))
    
    '''this has all feats ranked for all dev set subjects. need to identify:
        level of shared-ness
        https://liveastonac-my.sharepoint.com/personal/campelof_aston_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fcampelof%5Faston%5Fac%5Fuk%2FDocuments%2FMicrosoft%20Teams%20Chat%20Files%2F00%5F2%5FSupplementary%5FFile%5F6%2Epdf&parent=%2Fpersonal%2Fcampelof%5Faston%5Fac%5Fuk%2FDocuments%2FMicrosoft%20Teams%20Chat%20Files&ga=1
        plot rank by category as per the above
        
        for aggregate measure, could eg plot occurance rate of selected feats?
        # occurance against ranked occurance.
        or THEN merge the lag windows, plot occurance of selected against category or something.
    '''
    
    if 0:
           if 0:     
                '''
                tsne_eeg = TSNE(n_components)
    
                print('starting eeg tsne')
                tsne_result_eeg = tsne_eeg.fit_transform(X_eeg)
                print(tsne_result_eeg.shape)            
                plot_tsne(tsne_result_eeg,y_eeg,('3% 2-30Hz eeg ppt '+str(idx)))
                '''
                
    if 0:          
            '''
            else:
                eeg_sessions_masks=get_session_masks(eeg_ppt)
                for idx2,eeg_session_mask in enumerate(eeg_sessions_masks):
                    eeg=eeg_ppt[eeg_session_mask]
                    y_eeg=eeg.pop('Label')
                    X_eeg=eeg
                    X_eeg=ml.drop_ID_cols(X_eeg)
                    X_eeg,eegscaler=feats.scale_feats_train(X_eeg,'standardise')
                
                    X_eeg['Label']=y_eeg
                    sel_cols_eeg=feats.sel_feats_l1_df(X_eeg,sparsityC=0.005,maxfeats=40)
                    X_eeg.pop('Label')
                    X_eeg=X_eeg.iloc[:,sel_cols_eeg]
                
                    tsne_eeg = TSNE(n_components)
        
                    print('starting eeg tsne')
                    tsne_result_eeg = tsne_eeg.fit_transform(X_eeg)
                    print(tsne_result_eeg.shape)            
                    plot_tsne(tsne_result_eeg,y_eeg,('CSP eeg ppt '+str(idx)+' session '+str(idx2)))
                '''