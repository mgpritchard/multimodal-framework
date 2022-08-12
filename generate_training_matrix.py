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
from live_feature_extraction import generate_feature_vectors_from_samples, generate_feature_vectors_from_samples_single
import importlib.util
spec = importlib.util.spec_from_file_location("toClass", params.gen_trainmat_spec_SpellLoc)
toClass = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toClass)
'''INVESTIGATE IF THIS IS THE RIGHT TOCLASS VER. SHOULD PROB HAVE ONE IN HERE?'''
#foo.MyClass()


def gen_training_matrix(directory_path, output_file, cols_to_ignore, singleFrame=1):
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
    
    for x in os.listdir(directory_path):
        

        # Ignore non-CSV files
        if not x.lower().endswith('.csv'):
            continue
        
        # For safety we'll ignore files containing the substring "test". 
        # [Test files should not be in the dataset directory in the first place]
        if 'test' in x.lower():
            continue
        try:
            name, state, _ = x[:-4].split('-')
        except:
            print ('Wrong file name', x)
            sys.exit(-1)
        '''
        if state.lower() == 'open':
            state = 0.
        elif state.lower() == 'neutral':
            state = 1.
        elif state.lower() == 'close':
            state = 2.
        elif state.lower() == 'grasp':
            state = 3.
        elif state.lower() == 'lateral':
            state = 4.
        elif state.lower() == 'tripod':
            state = 5.
        else:
            print ('Wrong file name', x)
            print(state.lower())
            sys.exit(-1)
        '''
        try:
            state = params.gestures_to_idx[state.lower()]
        except KeyError:
            print ('Wrong file name', x)
            print(state.lower())
            sys.exit(-1)
       
        print ('Using file', x)
        full_file_path = directory_path  +   '/'   + x            
        if not singleFrame:
            vectors, header = generate_feature_vectors_from_samples(file_path = full_file_path, 
                                                                nsamples = 150, 
                                                                #period = 1, #it would be 1 for unicorn eeg which is 1234567890.123456
                                                                period=1000, #it would be 1000 for myo emg which is  1234567890123.4
                                                                state = state,
                                                                remove_redundant = True,
                                                                cols_to_ignore = cols_to_ignore)
        else:
            vectors, header = generate_feature_vectors_from_samples_single(file_path = full_file_path, 
                                                                nsamples = 150, 
                                                                #period = 1,
                                                                period=1000,
                                                                state = state,
                                                                remove_redundant = False,
                                                                cols_to_ignore = cols_to_ignore)
        
        print ('resulting vector shape for the file', vectors.shape)
        
        
        if FINAL_MATRIX is None:
            FINAL_MATRIX = vectors
        else:
            if np.any(vectors):
                FINAL_MATRIX = np.vstack( [ FINAL_MATRIX, vectors ] )
            else:
                print('got nothing back for '+x+', skipping')

    print ('FINAL_MATRIX', FINAL_MATRIX.shape)
        
    # Shuffle rows
    np.random.shuffle(FINAL_MATRIX)
    
    # Save to file
    np.savetxt(output_file, FINAL_MATRIX, delimiter = ',',
            header = ','.join(header), 
            comments = '')
    
    #Labelled_FINAL_MAT = FINAL_MATRIX.copy()[:,:-1]
    row_labels=[]
    for count, label in enumerate(FINAL_MATRIX[:,-1]):
        row_labels.append(params.idx_to_gestures[label])
    #Labelled_FINAL_MAT=np.concatenate((Labelled_FINAL_MAT,np.transpose(np.array([row_labels]))),axis=1)
    MatFrame=pd.DataFrame(FINAL_MATRIX.copy()[:,:-1],columns=header[:-1])
    MatFrame['Label']=row_labels
    MatFrame.to_csv((output_file[:-4] + '_Labelled.csv'),sep=',',index=False,float_format="%.18e")

    #https://phychai.wordpress.com/2016/10/27/pandas-to_csv-precision-loss/

    print (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),'Done.')
    
    return None


if __name__ == '__main__':
    """
    Main function. The parameters for the script are the following:
        [1] directory_path: The directory where the script will look for the files to process.
        [2] output_file: The filename of the generated output file.
    
    ATTENTION: It will ignore the last column of the CSV file by default
            (ie if cols_to_ignore remains -1). 
    
    Author:
        Original by [lmanso]
        Documentation: [fcampelo]
"""
    if len(sys.argv) < 3:
        print ('arg1: input dir\narg2: output file')
        sys.exit(-1)
    directory_path = sys.argv[1]
    output_file = sys.argv[2]
    gen_training_matrix(directory_path, output_file, cols_to_ignore = None)
    labelled_file=output_file[:-4]+'Class.csv'
    toClass.numtoClass(output_file,labelled_file)
