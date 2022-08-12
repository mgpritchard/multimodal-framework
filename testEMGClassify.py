#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 01:35:12 2022

@author: pritcham
"""

import numpy as np
import handleDatawrangle as wrangle
import handleFeats as feats
import handleML as ml
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename


#run handleComposeDataset

#here=os.path.dirname(os.path.realpath(__file__))
root="/home/michael/Documents/Aston/MultimodalFW/working_dataset/"
Tk().withdraw()

'''
datain=askdirectory(initialdir=root,title='Select EMG Directory')
dataout=datain
wrangle.process_emg(datain,dataout)
print('**processed raw emg**')
#sync_raw_files()    #not yet ready as EEG data not ready
emg_data_path=dataout
emg_feats_file=asksaveasfilename(initialdir=root,title='Save featureset as')
feats.make_feats(directory_path=emg_data_path,output_file=emg_feats_file,datatype='emg')
print('**made featureset**')

featset=ml.matrix_from_csv_file(emg_feats_file)[0]
print('**loaded featureset**')
#split the dataset into train and test
train,test=ml.skl.model_selection.train_test_split(featset,test_size=0.25,shuffle=False)
print('**split train/test**')

train_path=emg_feats_file[:-4]+'_trainslice.csv'
test_path=emg_feats_file[:-4]+'_testslice.csv'
np.savetxt(train_path, train, delimiter = ',')
np.savetxt(test_path, test, delimiter = ',')
print('**saved train/test splits')

ml.train_offline('RF',train_path)
print('**trained a model**')
'''

if not 'test' in locals():
    testset_loc=askopenfilename(initialdir=root,title='Select test set')
    test=ml.matrix_from_csv_file(testset_loc)[0]
testset_values = test[:,0:-1]
testset_labels = test[:,-1]
model = ml.load_model('testing emg',root)
#labels = list of labels in the order the distro is in
labels=model.classes_
distrolist=[]
predlist=[]
correctness=[]
for inst_count, instance in enumerate(testset_values):
    distro=ml.prob_dist(model, instance.reshape(1,-1))  
    predlabel=ml.pred_from_distro(labels, distro)
    distrolist.append(distro)
    predlist.append(predlabel)
    if predlabel == testset_labels[inst_count]:
        correctness.append(True)
    else:
        correctness.append(False)
accuracy = sum(correctness)/len(correctness)
print(accuracy)



