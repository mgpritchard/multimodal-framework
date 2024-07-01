#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
## Version history:

2018:
    Original script by Dr. Luis Manso [lmanso], Aston University
    
2019, June:
    Revised, commented and updated by Dr. Felipe Campelo [fcampelo], Aston University
    (f.campelo@aston.ac.uk / fcampelo@gmail.com)
    
2020 - 2022: tweaks made by Michael Pritchard [pritcham]

"""

import os, sys
import time
import numpy as np
import pandas as pd
import params
import traceback
#from live_feature_extraction import generate_feature_vectors_from_samples
from deepL_raw_extracts import generate_raw_samples



def gen_training_matrix(directory_path, output_file, cols_to_ignore, period=1000,auto_skip_all_fails=False):
    """
    Reads the csv files in directory_path and assembles the training matrix with 
    the features extracted using the functions from EEG_feature_extraction.
    
    Parameters:
        directory_path (str): directory containing the CSV files to process.
        output_file (str): filename for the output file.
        cols_to_ignore (list): list of columns to ignore from the CSV

    Returns:
        numpy.ndarray: 2D matrix containing the data read from the CSV
    
    Author: 
        Original: [lmanso] 
        Updates and documentation: [fcampelo]
    """
    
    # Initialise return matrix
    FINAL_MATRIX = None
    rejected_by_feat_script=[]
    
    for x in os.listdir(directory_path):
        

        # Ignore non-CSV files
        if not x.lower().endswith('.csv'):
            continue
        
        # For safety we'll ignore files containing the substring "test". 
        # [Test files should not be in the dataset directory in the first place]
        if 'test' in x.lower():
            continue
        try:
            name, state, count = x[:-4].split('-')
        except:
            if x[-7:-4]=='EEG':
                try:
                    name, state, count = x[:-9].split('-')
                except:
                    print ('Wrong file name', x)
                    sys.exit(-1)
            else:
                print ('Wrong file name', x)
                sys.exit(-1)

        try:
            state = params.gestures_to_idx[state.lower()]
        except KeyError:
            print ('Wrong file name', x)
            print(state.lower())
            sys.exit(-1)
       
        print ('Using file', x)
        full_file_path = directory_path  +   '/'   + x    

        
        try:

            vectors, header = generate_raw_samples(file_path = full_file_path, 
                                                                nsamples = 150, 
                                                                period=period,
                                                                state = state,
                                                                cols_to_ignore = cols_to_ignore)

        except UnboundLocalError as e:
            print(traceback.format_exc())
            if auto_skip_all_fails==True:
                print('Skipping the data file:\n'+full_file_path)
                rejected_by_feat_script.append(full_file_path)
                continue
            else:
                skip=input('The above error was encountered when trying to generate '
                      'features from the following data file:\n'+full_file_path+
                      '\nWould you like to skip the file and continue? [Y/N]')
                if skip == 'Y':
                    rejected_by_feat_script.append(full_file_path)
                    continue
                elif skip == 'y':
                    rejected_by_feat_script.append(full_file_path)
                    continue
                else:
                    raise e
        print ('resulting vector shape for the file', vectors.shape)
        
        
        if FINAL_MATRIX is None:
            FINAL_MATRIX = vectors
        else:
            if np.any(vectors):
                FINAL_MATRIX = pd.concat( [ FINAL_MATRIX, vectors ] , axis = 0)
            else:
                print('got nothing back for '+x+', skipping')

    print ('FINAL_MATRIX', FINAL_MATRIX.shape)
    
    
    
    
    
    
    
    
    # Shuffle rows
    #np.random.shuffle(FINAL_MATRIX)
    FINAL_MATRIX=FINAL_MATRIX.sample(frac=1).reset_index(drop=True)
    
    # Save to file
    '''
    np.savetxt(output_file, FINAL_MATRIX, delimiter = ',',
            header = ','.join(header), 
            comments = '')
    '''
    
    FINAL_MATRIX.to_pickle(output_file)
    #https://stackoverflow.com/questions/17098654/how-to-reversibly-store-and-load-a-pandas-dataframe-to-from-disk
    
    if rejected_by_feat_script:
        np.savetxt(output_file[:-4]+'_REJECTED.csv',rejected_by_feat_script,
               delimiter=',', fmt="%s")
    
    '''
    #Labelled_FINAL_MAT = FINAL_MATRIX.copy()[:,:-1]
    row_labels=[]
    for count, label in enumerate(FINAL_MATRIX[:,-1]):
        row_labels.append(params.idx_to_gestures[label])
    #Labelled_FINAL_MAT=np.concatenate((Labelled_FINAL_MAT,np.transpose(np.array([row_labels]))),axis=1)
    MatFrame=pd.DataFrame(FINAL_MATRIX.copy()[:,:-1],columns=header[:-1])
    MatFrame['Label']=row_labels
    MatFrame.to_csv((output_file[:-4] + '_Labelled.csv'),sep=',',index=False,float_format="%.18e")
    '''
    
    Labelled_Matrix=FINAL_MATRIX.copy()
    Labelled_Matrix['Label']=Labelled_Matrix['Label'].map(params.idx_to_gestures)
    #https://stackoverflow.com/questions/57350896/use-a-python-dictionary-as-a-lookup-table-to-output-new-values
    Labelled_Matrix.to_pickle((output_file[:-4] + '_Labelled.pkl'))

    #https://phychai.wordpress.com/2016/10/27/pandas-to_csv-precision-loss/

    print (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),'Done.')
    
    return FINAL_MATRIX



if __name__ == '__main__':
    """
    Main function. The parameters for the script are the following:
        [1] directory_path: The directory where the script will look for the files to process.
        [2] output_file: The filename of the generated output file.
    
    
    Author:
        Original by [lmanso]
        Documentation: [fcampelo]
"""
    if len(sys.argv) > 2:
        print ('arg1: input dir\narg2: output file')
        #sys.exit(-1)
        directory_path = sys.argv[1]
        output_file = sys.argv[2]
        
    else:
        #directory_path=r"H:\Jeong11tasks_data\deepLcompare\dummy_EMG"
        directory_path=r"H:\Jeong11tasks_data\EMG\Raw_EMGCSVs"
        #output_file=r"H:\Jeong11tasks_data\deepLcompare\trial_raw_EMG.pkl"
        output_file=r"H:\Jeong11tasks_data\deepLcompare\all_raw_EMG.pkl"
        
    gen_training_matrix(directory_path, output_file, cols_to_ignore = None)

