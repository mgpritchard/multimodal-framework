# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 20:19:39 2023

@author: pritcham
"""
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

bespoke_ho_means=StringIO("""arch ppt fusion_acc
just_emg ppt1 0.8032
just_emg ppt6 0.8249
just_emg ppt11 0.9455
just_emg ppt16 0.8376
just_emg ppt21 0.865
decision ppt1 0.8125
decision ppt6 0.8144
decision ppt11 0.9467
decision ppt16 0.8292
decision ppt21 0.8661
feat_sep ppt1 0.7812
feat_sep ppt6 0.8170
feat_sep ppt11 0.8980
feat_sep ppt16 0.7891
feat_sep ppt21 0.8287
feat_join ppt1 0.7747
feat_join ppt6 0.8101
feat_join ppt11 0.8960
feat_join ppt16 0.7816
feat_join ppt21 0.8114
hierarch ppt1 0.8293
hierarch ppt6 0.8345
hierarch ppt11 0.9467
hierarch ppt16 0.8324
hierarch ppt21 0.8697
inv_hierarch ppt1 0.7518
inv_hierarch ppt6 0.7942
inv_hierarch ppt11 0.8900
inv_hierarch ppt16 0.7619
inv_hierarch ppt21 0.7907
""")

df=pd.read_table(bespoke_ho_means,sep=" ")


_,acc_noEEG=plt.subplots()
#scores1.loc[~scores1.index.isin(['just_eeg'],level='arch')].plot(y='fusion_acc',ax=acc_noEEG)#,style='o')
#scores6.loc[~scores1.index.isin(['just_eeg'],level='arch')].plot(y='fusion_acc',ax=acc_noEEG)#,style='o')
#scores11.loc[~scores1.index.isin(['just_eeg'],level='arch')].plot(y='fusion_acc',ax=acc_noEEG)#,style='o')
#scores16.loc[~scores1.index.isin(['just_eeg'],level='arch')].plot(y='fusion_acc',ax=acc_noEEG)#,style='o')
#scores21.loc[~scores1.index.isin(['just_eeg'],level='arch')].plot(y='fusion_acc',ax=acc_noEEG)#,style='o')
df[df['ppt']=='ppt1'].plot(y='fusion_acc',x='arch',ax=acc_noEEG)
df[df['ppt']=='ppt6'].plot(y='fusion_acc',x='arch',ax=acc_noEEG)
df[df['ppt']=='ppt11'].plot(y='fusion_acc',x='arch',ax=acc_noEEG)
df[df['ppt']=='ppt16'].plot(y='fusion_acc',x='arch',ax=acc_noEEG)
df[df['ppt']=='ppt21'].plot(y='fusion_acc',x='arch',ax=acc_noEEG)

acc_noEEG.get_figure().suptitle('')
acc_noEEG.set_title('')
acc_noEEG.legend(['ppt1','ppt6','ppt11','ppt16','ppt21'])
acc_noEEG.set(xlabel='Architecture')