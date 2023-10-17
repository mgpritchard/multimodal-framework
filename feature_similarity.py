# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 18:05:48 2023

@author: pritcham
"""


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


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

def overlap(list1,list2):
    intersection=len(list(set(list1).intersection(list2)))
    return intersection/min(len(list1),len(list2))

bespokepath=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\datasets\shared_feat_analysis\eeg-Bespoke-devsetOnly-feats.csv"
generalistpath=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\datasets\shared_feat_analysis\eeg-Generalist-devsetOnly-feats.csv"

bespokeFeats=pd.read_csv(bespokepath,header=None,index_col=False)
generalistFeats=pd.read_csv(generalistpath,header=None,index_col=False)
generalistFeats=generalistFeats[~generalistFeats.isna().any(axis=1)].reset_index(drop=True)

bespoke_choices_unique=bespokeFeats.T.copy().applymap(lambda x: x.replace('lag1_',''))
bespoke_choices_unique=bespoke_choices_unique.apply(lambda x: x[~x.duplicated()]).fillna('')

bespoke_popular_unique=bespoke_choices_unique.stack().value_counts().drop(index='')
bespoke_populars=bespoke_popular_unique.rename_axis('feat').reset_index()
bespoke_populars['category']=bespoke_populars['feat'].copy()
bespoke_populars['category']=bespoke_populars['category'].apply(lambda x: x[:min([i for i, c in enumerate(x) if c=='_' and not any(char.isalpha() for char in x[i+1:])])])
bespoke_populars=bespoke_populars.rename(columns={0:'occurence'})
bespoke_populars['electrode']=[int(x.split('_')[-1]) if not ('covM' in x or 'eigen' in x) else None for x in bespoke_populars['feat']]
print('0 - 5 and 7,8,9 = LHS, 11 - 19 = RHS, 6 & 10 = central')
bespoke_populars['hemisphere']=['central' if (x==6 or x==10) else 'left' if x<10 else 'right' if x>10 else None for x in bespoke_populars['electrode']]
bespoke_populars['rank']=len(bespoke_populars)-bespoke_populars['occurence'].rank(method='first')


generalist_choices_unique=generalistFeats.T.copy().applymap(lambda x: x.replace('lag1_',''))
generalist_choices_unique=generalist_choices_unique.apply(lambda x: x[~x.duplicated()]).fillna('')

generalist_popular_unique=generalist_choices_unique.stack().value_counts().drop(index='')
generalist_populars=generalist_popular_unique.rename_axis('feat').reset_index()
generalist_populars['category']=generalist_populars['feat'].copy()
generalist_populars=generalist_populars.loc[generalist_populars['category']!='Label']
generalist_populars['category']=generalist_populars['category'].apply(lambda x: x[:min([i for i, c in enumerate(x) if c=='_' and not any(char.isalpha() for char in x[i+1:])])])
generalist_populars=generalist_populars.rename(columns={0:'occurence'})
generalist_populars['electrode']=[int(x.split('_')[-1]) if not ('covM' in x or 'eigen' in x) else None for x in generalist_populars['feat']]
generalist_populars['hemisphere']=['central' if (x==6 or x==10) else 'left' if x<10 else 'right' if x>10 else None for x in generalist_populars['electrode']]
generalist_populars['rank']=len(generalist_populars)-generalist_populars['occurence'].rank(method='first')

generalist_populars['broad_category']=[broad_cats[x] for x in generalist_populars['category']]

plt.figure()
#populars=populars.sort_values('category')
sns.scatterplot(x='rank', y='occurence', data=bespoke_populars[bespoke_populars['occurence']>1], hue='hemisphere', ec=None,palette='muted')
print(bespoke_populars['hemisphere'].value_counts())

plt.figure()
#populars=populars.sort_values('category')
sns.scatterplot(x='rank', y='occurence', data=generalist_populars[generalist_populars['occurence']>1], hue='hemisphere', ec=None,palette='muted')
print(generalist_populars['hemisphere'].value_counts())


plt.figure()
generalist_populars=generalist_populars.sort_values('broad_category')
sns.scatterplot(x='rank', y='occurence', data=generalist_populars[generalist_populars['occurence']>9], hue='category', ec=None,palette='muted')
print(generalist_populars['category'].value_counts())
plt.figure()
sns.scatterplot(x='rank', y='occurence', data=generalist_populars[generalist_populars['occurence']>5], hue='broad_category', ec=None,palette='muted')
print(generalist_populars['broad_category'].value_counts())
plt.figure()
sns.scatterplot(x='rank', y='occurence', data=generalist_populars[generalist_populars['occurence']>14], hue='feat', ec=None,palette='Set2')
#plt.figure()
#sns.scatterplot(x='rank', y='occurence', data=generalist_populars[generalist_populars['occurence']>19], hue='feat', ec=None,palette='Set2')

plt.scatter(generalist_populars['rank'],generalist_populars['occurence'],c=pd.factorize(generalist_populars['category'])[0],cmap='Dark2',label=generalist_populars['category'].unique())
plt.legend(pd.factorize(generalist_populars['category'])[0]) 



bespoke_quarterplus=bespoke_populars.loc[bespoke_populars['occurence']>5]
bespoke_qplus_list=bespoke_quarterplus['feat'].tolist()
bespoke_3quarterplus=bespoke_populars.loc[bespoke_populars['occurence']>14]
bespoke_3qplus_list=bespoke_3quarterplus['feat'].tolist()
generalist_quarterplus=generalist_populars.loc[generalist_populars['occurence']>5]
generalist_qplus_list=generalist_quarterplus['feat'].tolist()
generalist_3quarterplus=generalist_populars.loc[generalist_populars['occurence']>14]
generalist_3qplus_list=generalist_3quarterplus['feat'].tolist()
print('Overlap between popular (more than quarter chosen) Bespoke & Generalist EEG features: ',overlap(bespoke_qplus_list,generalist_qplus_list))
print('Overlap between quaterplus-Bespoke & 3quartsplus-Generalist EEG features: ',overlap(bespoke_qplus_list,generalist_3qplus_list))
print('Overlap between 3qplus-Bespoke & 3quartsplus-Generalist EEG features: ',overlap(bespoke_3qplus_list,generalist_3qplus_list))
raise

popular_generalist=generalist_choices_unique.stack().value_counts().drop(index='')
generalist_overhalf=popular_generalist[popular_generalist>10]
generalist_overhalf=generalist_overhalf.index.tolist()
generalist_overquarter=popular_generalist[popular_generalist>5]
''' go to line 214 in rank_feats_bespoke! '''


featJoinGenpath=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\datasets\shared_feat_analysis\featfuseJoinsel-Generalist-devsetOnly-feats.csv"
featJoin_Gen_feats=pd.read_csv(featJoinGenpath,header=None,index_col=False)
featJoin_Gen_feats=featJoin_Gen_feats[~featJoin_Gen_feats.isna().any(axis=1)].reset_index(drop=True)

featJoin_Gen_choices_unique=featJoin_Gen_feats.T.copy().applymap(lambda x: x.replace('lag1_',''))
featJoin_Gen_choices_unique=featJoin_Gen_choices_unique.apply(lambda x: x[~x.duplicated()]).fillna('')

popular_featJoin_generalist=featJoin_Gen_choices_unique.stack().value_counts().drop(index='')
''' go to line 214 in rank_feats_bespoke! '''
featJoin_generalist_populars=popular_featJoin_generalist.rename_axis('feat').reset_index()
featJoin_generalist_populars['category']=featJoin_generalist_populars['feat'].copy()
featJoin_generalist_populars=featJoin_generalist_populars.loc[featJoin_generalist_populars['category']!='Label']
featJoin_generalist_populars['category']=featJoin_generalist_populars['category'].apply(lambda x: x[:min([i for i, c in enumerate(x) if c=='_' and not any(char.isalpha() for char in x[i+1:])])])
featJoin_generalist_populars=featJoin_generalist_populars.rename(columns={0:'occurence'})

featJoin_generalist_populars['EEGfeats']=[x[4:] if 'EEG_' in x else None for x in featJoin_generalist_populars['feat']]
featJoin_generalist_populars['rank']=len(featJoin_generalist_populars)-featJoin_generalist_populars['occurence'].rank(method='first')

plt.figure()
sns.scatterplot(x='rank', y='occurence', data=featJoin_generalist_populars.loc[(featJoin_generalist_populars['EEGfeats'].notna()) & (featJoin_generalist_populars['occurence']>9)], hue='category', ec=None,palette='muted')
print((featJoin_generalist_populars.loc[featJoin_generalist_populars['EEGfeats'].notna()])['category'].value_counts())
plt.figure()
sns.scatterplot(x='rank', y='occurence', data=featJoin_generalist_populars[featJoin_generalist_populars['occurence']>14], hue='EEGfeats', ec=None,palette='Set2')
print(featJoin_generalist_populars['EEGfeats'].value_counts())

featJoin_gen_EEG_unique=(featJoin_generalist_populars.loc[featJoin_generalist_populars['EEGfeats'].notna()])
featJoin_gen_EEG_list=(featJoin_gen_EEG_unique.loc[featJoin_gen_EEG_unique['occurence']>10])['EEGfeats'].tolist()
generalist_EEG_list=generalist_popular_unique[generalist_popular_unique>10].index.tolist()
if 'Label' in featJoin_gen_EEG_list:
    featJoin_gen_EEG_list.remove('Label')
if 'Label' in generalist_EEG_list:
    generalist_EEG_list.remove('Label')
overlap_coef=overlap(generalist_EEG_list,featJoin_gen_EEG_list)
print('Overlap coefficient between halfplus generalist and halfplus featJoin generalist: ',overlap_coef)





generalistEMGpath=r"C:\Users\pritcham\Documents\mm-framework\multimodal-framework\lit_data_expts\jeong\datasets\shared_feat_analysis\emg-Generalist-devsetOnly-feats.csv"
generalistEMGFeats=pd.read_csv(generalistEMGpath,header=None,index_col=False)
generalistEMGFeats=generalistEMGFeats[~generalistEMGFeats.isna().any(axis=1)].reset_index(drop=True)
'''dont need the populars here, only using to join with EEG for checking Feat Fuse Sep/Joint Sel'''
'''for those purposes may actually want to include the lag feats, that way sets will be same size'''

for i in range(20):
    bespokes=bespoke_choices_unique.iloc[:,i]
    generalists=generalist_choices_unique.iloc[:,i]
    
    
    bespokes=list(filter(None,bespokes.tolist()))
    generalists=list(filter(None,generalists.tolist()))
    if 'Label' in bespokes:
        bespokes.remove('Label')
    if 'Label' in generalists:
        generalists.remove('Label')
    
    overlap_coef=overlap(bespokes,generalists)
    print('For dev set ',str(i),'overlap coef between EEG bespoke & generalist = ',str(overlap_coef))
    
    overlap_coef=overlap(bespokes,generalist_overhalf)
    print('overlap coef between EEG bespoke & generalist popular choices = ',str(overlap_coef))


#for i in range(20):
    sep_eeg=generalistFeats.iloc[i].tolist()
    sep_emg=generalistEMGFeats.iloc[i].tolist()
    join_sels=featJoin_Gen_feats.iloc[i].tolist()
    
    if 'Label' in sep_eeg:
        sep_eeg.remove('Label')
    if 'Label' in sep_emg:
        sep_emg.remove('Label')
    if 'Label' in join_sels:
        join_sels.remove('Label')
        
    sep_eeg=['EEG_' + s for s in sep_eeg]
    sep_sels=sep_emg+sep_eeg
    overlap_coef=overlap(sep_sels,join_sels)
    print('For dev set ',str(i),'overlap coef between Generalist FeatFuse Sep sel & Join sel = ',str(overlap_coef))
    
    '''now see how many of the EEG picks specifically showed up maybe??'''
    
    generalists_eeg=['EEG_' + s for s in generalists]
    generalists_eeg=sep_eeg
    #overlap_coef=overlap(generalists_eeg,sep_sels)
    #print('For dev set ',str(i),'overlap coef between EEG Gen & Feat Sep Gen = ',str(overlap_coef))
    overlap_coef=overlap(generalists_eeg,join_sels)
    print('For dev set ',str(i),'overlap coef between EEG Gen & Feat Join Gen = ',str(overlap_coef))





