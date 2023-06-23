# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 20:22:57 2023

@author: pritcham
"""
import pandas as pd
import numpy as np

 
csvpath=r'C:/Users/pritcham/Desktop/L1_top10.csv'
csvpath=r'C:/Users/pritcham/Desktop/L1_GENERALIST_top20.csv'
top10=pd.read_csv(open(csvpath,'rb'),index_col=False,header=None).T
top10_unique=top10.copy()
top10_unique.drop(0,axis=0,inplace=True)
top10_unique=top10_unique.applymap(lambda x: x.replace('lag1_',''))
top10_unique=top10_unique.apply(lambda x: x[~x.duplicated()]).fillna('')
popular_unique=top10_unique.stack().value_counts().drop(index='')



delta_0=top10_unique.applymap(lambda x: 'sum_delta_0' in x).sum(axis=0)
delta_3=top10_unique.applymap(lambda x: 'sum_delta_3' in x).sum(axis=0)
delta_13=top10_unique.applymap(lambda x: 'sum_delta_13' in x).sum(axis=0)
delta_16=top10_unique.applymap(lambda x: 'sum_delta_16' in x).sum(axis=0)
delta0ppts=delta_0[delta_0==1].index.values
delta3ppts=delta_3[delta_3==1].index.values
delta13ppts=delta_13[delta_13==1].index.values
delta16ppts=delta_16[delta_16==1].index.values
overlap_del0del3=np.intersect1d(delta0ppts,delta3ppts)
total_del0ordel3=np.unique(np.concatenate((delta0ppts,delta3ppts)))
overlap_del13del16=np.intersect1d(delta13ppts,delta16ppts)
total_del13ordel16=np.unique(np.concatenate((delta13ppts,delta16ppts)))
total_deltaLeftAndRight=np.intersect1d(total_del0ordel3,total_del13ordel16)
overlap_all=np.intersect1d(overlap_del0del3,overlap_del13del16)
