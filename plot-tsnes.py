# -*- coding: utf-8 -*-
"""
Created on Tue May 23 22:01:44 2023

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


def plot_tsne(tsne_result,y,title,cmap='viridis'): #cmap jet or Dark2_r
    tsne_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
    fig, ax = plt.subplots()
    scatter=plt.scatter(x=tsne_df['tsne_1'], y=tsne_df['tsne_2'], c=tsne_df['label'], s=1,cmap=cmap)
    # s is a size of marker , could also 'cmap'=https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.set_title(title)
    leg=ax.legend(*scatter.legend_elements(),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    leg.texts[0].set_text('cyl')
    leg.texts[1].set_text('sph')
    leg.texts[2].set_text('lumb')
    leg.texts[3].set_text('rest')
    plt.show()
    return tsne_df

def get_session_masks(featset):
    masks=[featset['ID_run']== run for run in np.sort(featset['ID_run'].unique())] 
    return masks


def plot_mass_tsne_per_ppt():
    eeg_masks=fuse.get_ppt_split_flexi(eeg_set)
    for idx,eeg_mask in enumerate(eeg_masks):
        tsne_eeg_ppt = tsne_result_eeg[eeg_mask]
        targs_ppt=y_eeg[eeg_mask]
        plot_tsne(tsne_eeg_ppt,targs_ppt,('eeg L1 40 ppt'+str(idx)))

emg_set_path=params.jeong_EMGfeats,
eeg_set_path=params.jeong_noCSP_WidebandFeats,
#eeg_set_path=params.eeg_jeongSyncCSP_feats,
#eeg_set_path=params.eeg_jeong_feats,

emg_set_path=params.jeong_emg_noholdout,
eeg_set_path=params.jeong_eeg_noholdout,
            
emg_set=ml.pd.read_csv(emg_set_path[0],delimiter=',')
eeg_set=ml.pd.read_csv(eeg_set_path[0],delimiter=',')

per_ppt = False
plotemg=False
ploteeg=True
n_components = 2

plt.rcParams['figure.dpi']=150




eegCSP_set_path=params.eeg_jeongSyncCSP_feats
eegCSP=ml.pd.read_csv(eegCSP_set_path,delimiter=',')
eegCSP_noHO=eegCSP[~eegCSP['ID_pptID'].isin([1,6,11,16,21])]

eeg_set_noCSP=eeg_set.copy()
eeg_set=eegCSP_noHO


y_eeg=eeg_set.pop('Label')
X_eeg=eeg_set
X_eeg=ml.drop_ID_cols(X_eeg)
tsne_eeg = TSNE(n_components)
   
print('starting csp eeg tsne')
tsne_result_eeg = tsne_eeg.fit_transform(X_eeg)
print(tsne_result_eeg.shape)
plot_tsne(tsne_result_eeg,y_eeg,'csp eeg')

cspEEGtsne=(tsne_result_eeg,y_eeg)
plot_tsne(cspEEGtsne[0],cspEEGtsne[1],'Dev set EEG with CSP, no feat reduction')
pickle.dump(cspEEGtsne,open(r"C:\Users\pritcham\Documents\RQ1_plots_stats\newTSNEsNoHoldout\CSP_EEGnoHO.pkl",'wb'))




if per_ppt:
    emg_masks=fuse.get_ppt_split_flexi(emg_set)
    eeg_masks=fuse.get_ppt_split_flexi(eeg_set)
    for idx,emg_mask in enumerate(emg_masks):
        eeg_mask=eeg_masks[idx]
        split_sessions=False
        if plotemg:
            
            emg_ppt = emg_set[emg_mask]
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

        if ploteeg:
            eeg_ppt = eeg_set[eeg_mask]
            if not split_sessions:
                y_eeg=eeg_ppt.pop('Label')
                X_eeg=eeg_ppt
                X_eeg=ml.drop_ID_cols(X_eeg)
                X_eeg,eegscaler=feats.scale_feats_train(X_eeg,'standardise')
            
                X_eeg['Label']=y_eeg
                #sel_cols_eeg=feats.sel_percent_feats_df(X_eeg,percent=3)
                sel_cols_eeg=feats.sel_feats_l1_df(X_eeg,sparsityC=0.005,maxfeats=40)
                X_eeg.pop('Label')
                X_eeg=X_eeg.iloc[:,sel_cols_eeg]
            
                tsne_eeg = TSNE(n_components)
    
                print('starting eeg tsne')
                tsne_result_eeg = tsne_eeg.fit_transform(X_eeg)
                print(tsne_result_eeg.shape)            
                plot_tsne(tsne_result_eeg,y_eeg,('L1 40 2-30Hz eeg ppt '+str(idx)))
                print('Dont have 2-30Hz CSP to compare tho...')
            else:
                eeg_sessions_masks=get_session_masks(eeg_ppt)
                for idx2,eeg_session_mask in enumerate(eeg_sessions_masks):
                    eeg=eeg_ppt[eeg_session_mask]
                    y_eeg=eeg.pop('Label')
                    X_eeg=eeg
                    X_eeg=ml.drop_ID_cols(X_eeg)
                    X_eeg,eegscaler=feats.scale_feats_train(X_eeg,'standardise')
                
                    X_eeg['Label']=y_eeg
                    #sel_cols_eeg=feats.sel_percent_feats_df(X_eeg,percent=15)
                    sel_cols_eeg=feats.sel_feats_l1_df(X_eeg,sparsityC=0.005,maxfeats=40)
                    X_eeg.pop('Label')
                    X_eeg=X_eeg.iloc[:,sel_cols_eeg]
                
                    tsne_eeg = TSNE(n_components)
        
                    print('starting eeg tsne')
                    tsne_result_eeg = tsne_eeg.fit_transform(X_eeg)
                    print(tsne_result_eeg.shape)            
                    plot_tsne(tsne_result_eeg,y_eeg,('CSP eeg ppt '+str(idx)+' session '+str(idx2)))

else:
    tsne_raw=True
    if plotemg:        
        y_emg=emg_set.pop('Label')
        X_emg=emg_set
        X_emg=ml.drop_ID_cols(X_emg)
        tsne_emg = TSNE(n_components)
        
        if tsne_raw:
            print('starting raw emg tsne')
            tsne_result_emg = tsne_emg.fit_transform(X_emg)
            print(tsne_result_emg.shape)
            plot_tsne(tsne_result_emg,y_emg,'raw emg')
            #tsne_result_emg_df = pd.DataFrame({'tsne_1': tsne_result_emg[:,0], 'tsne_2': tsne_result_emg[:,1], 'label': y_emg})
        else:
            X_emg,emgscaler=feats.scale_feats_train(X_emg,'standardise')
            X_emg['Label']=y_emg
            sel_cols_emg=feats.sel_percent_feats_df(X_emg,percent=15)
            #sel_cols_emg=np.append(sel_cols_emg,X_emg.columns.get_loc('Label'))
            X_emg.pop('Label')
            X_emg=X_emg.iloc[:,sel_cols_emg]
            
            print('starting emg tsne')
            tsne_result_emg = tsne_emg.fit_transform(X_emg)
            print(tsne_result_emg.shape)
            
            plot_tsne(tsne_result_emg,y_emg,'emg 15%')
            
            
    
    if ploteeg:
        y_eeg=eeg_set.pop('Label')
        X_eeg=eeg_set
        X_eeg=ml.drop_ID_cols(X_eeg)
        tsne_eeg = TSNE(n_components)
    #emg_set,eeg_set=balance_set(emg_set,eeg_set)
    
    #https://danielmuellerkomorowska.com/2021/01/05/introduction-to-t-sne-in-python-with-scikit-learn/
        
        if tsne_raw:   
            print('starting raw eeg tsne')
            tsne_result_eeg = tsne_eeg.fit_transform(X_eeg)
            print(tsne_result_eeg.shape)
            plot_tsne(tsne_result_eeg,y_eeg,'raw eeg')
        else:
            X_eeg,eegscaler=feats.scale_feats_train(X_eeg,'standardise')
        
            X_eeg['Label']=y_eeg
            #sel_cols_eeg_old=feats.sel_percent_feats_df(X_eeg,percent=3)
            #sel_cols_eeg=feats.sel_feats_l1_df(X_eeg,sparsityC=0.01)
            sel_cols_eeg=feats.sel_feats_l1_df(X_eeg,sparsityC=0.005,maxfeats=40)
            #sel_cols_eeg=np.append(sel_cols_eeg,X_eeg.columns.get_loc('Label'))
            X_eeg.pop('Label') #may not be needed?
            X_eeg=X_eeg.iloc[:,sel_cols_eeg]

            print('starting eeg tsne')
            tsne_result_eeg = tsne_eeg.fit_transform(X_eeg)
            print(tsne_result_eeg.shape)
            
            plot_tsne(tsne_result_eeg,y_eeg,'eeg L1 40')
            plot_mass_tsne_per_ppt()

'''
rawEEGtsne=(tsne_result_eeg,y_eeg)
plot_tsne(rawEEGtsne[0],rawEEGtsne[1],'Dev set EEG without CSP, all features')
pickle.dump(rawEEGtsne,open(r"C:\Users\pritcham\Documents\RQ1_plots_stats\newTSNEsNoHoldout\rawEEGnoCSPnoHO.pkl",'wb'))
'''