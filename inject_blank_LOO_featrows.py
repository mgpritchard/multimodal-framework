# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 21:53:44 2023

@author: pritcham
"""

'''
script to stick blank rows into the LOO arrays of selected features.
ppts 1, 6, 11, 16, and 21 are held out (zero-indexed in the dataframe)
so when the LOO looks up features, it needs to be able to index properly by
participant ID and hence needs these blank rows as buffers
'''

emgfeats=pd.read_csv(params.emgLOOfeatpath,header=None)
nulls=['N/A']*89
nullrow=pd.DataFrame(np.array(nulls).reshape(-1,len(nulls)),columns=emgfeats.columns)
emgfeats2=pd.concat([emgfeats[:0],nullrow,emgfeats[0:]]).reset_index(drop=True)
emgfeats2=pd.concat([emgfeats2[:5],nullrow,emgfeats2[5:]]).reset_index(drop=True)
emgfeats2=pd.concat([emgfeats2[:10],nullrow,emgfeats2[10:]]).reset_index(drop=True)
emgfeats2=pd.concat([emgfeats2[:15],nullrow,emgfeats2[15:]]).reset_index(drop=True)
emgfeats2=pd.concat([emgfeats2[:20],nullrow,emgfeats2[20:]]).reset_index(drop=True)
emgfeats2.to_csv(params.emgLOOfeatpath,header=False,index=False)

eegfeats=pd.read_csv(params.eegLOOfeatpath,header=None)
eegfeats2=pd.concat([eegfeats[:0],nullrow,eegfeats[0:]]).reset_index(drop=True)
eegfeats2=pd.concat([eegfeats2[:5],nullrow,eegfeats2[5:]]).reset_index(drop=True)
eegfeats2=pd.concat([eegfeats2[:10],nullrow,eegfeats2[10:]]).reset_index(drop=True)
eegfeats2=pd.concat([eegfeats2[:15],nullrow,eegfeats2[15:]]).reset_index(drop=True)
eegfeats2=pd.concat([eegfeats2[:20],nullrow,eegfeats2[20:]]).reset_index(drop=True)
eegfeats2.to_csv(params.eegLOOfeatpath,header=False,index=False)

jointfeats=pd.read_csv(params.jointemgeegLOOfeatpath,header=None)
nulls=['N/A']*177
nullrow=pd.DataFrame(np.array(nulls).reshape(-1,len(nulls)),columns=jointfeats.columns)
jointfeats2=pd.concat([jointfeats[:0],nullrow,jointfeats[0:]]).reset_index(drop=True)
jointfeats2=pd.concat([jointfeats2[:5],nullrow,jointfeats2[5:]]).reset_index(drop=True)
jointfeats2=pd.concat([jointfeats2[:10],nullrow,jointfeats2[10:]]).reset_index(drop=True)
jointfeats2=pd.concat([jointfeats2[:15],nullrow,jointfeats2[15:]]).reset_index(drop=True)
jointfeats2=pd.concat([jointfeats2[:20],nullrow,jointfeats2[20:]]).reset_index(drop=True)
jointfeats2.to_csv(params.jointemgeegLOOfeatpath,header=False,index=False)



emgfeats=pd.read_csv(params.emgLOOfeatpath,header=None)
emgfeats_20=pd.read_csv(r"C:\Users\pritcham\Desktop\emg_feat20.csv",header=None).T
#nulls=['N/A']*89
#nullrow=pd.DataFrame(np.array(nulls).reshape(-1,len(nulls)),columns=emgfeats.columns)
emgfeats.loc[0]=emgfeats_20.loc[0]
emgfeats.loc[5]=emgfeats_20.loc[0]
emgfeats.loc[10]=emgfeats_20.loc[0]
emgfeats.loc[15]=emgfeats_20.loc[0]
emgfeats.loc[20]=emgfeats_20.loc[0]
emgfeats.to_csv(params.emgLOOfeatpath,header=False,index=False)

eegfeats=pd.read_csv(params.eegLOOfeatpath,header=None)
eegfeats_20=pd.read_csv(r"C:\Users\pritcham\Desktop\eeg_feat20.csv",header=None).T
#nulls=['N/A']*89
#nullrow=pd.DataFrame(np.array(nulls).reshape(-1,len(nulls)),columns=emgfeats.columns)
eegfeats.loc[0]=eegfeats_20.loc[0]
eegfeats.loc[5]=eegfeats_20.loc[0]
eegfeats.loc[10]=eegfeats_20.loc[0]
eegfeats.loc[15]=eegfeats_20.loc[0]
eegfeats.loc[20]=eegfeats_20.loc[0]
eegfeats.to_csv(params.eegLOOfeatpath,header=False,index=False)

jointfeats=pd.read_csv(params.jointemgeegLOOfeatpath,header=None)
'''Now rename the one you just read to xyz_HoldoutdsBlank.csv or similar!'''
joinfeats_20=pd.read_csv(r"C:\Users\pritcham\Desktop\join_feat20.csv",header=None).T
jointfeats.loc[0]=joinfeats_20.loc[0]
jointfeats.loc[5]=joinfeats_20.loc[0]
jointfeats.loc[10]=joinfeats_20.loc[0]
jointfeats.loc[15]=joinfeats_20.loc[0]
jointfeats.loc[20]=joinfeats_20.loc[0]
jointfeats.to_csv(params.jointemgeegLOOfeatpath,header=False,index=False)
