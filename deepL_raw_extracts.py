#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
## Version history:

2018:
    Original script by Dr. Luis Manso [lmanso], Aston University
    
2019, June:
    Revised, commented and updated by Dr. Felipe Campelo [fcampelo], Aston University
    (f.campelo@aston.ac.uk / fcampelo@gmail.com)
    
2020 - 2021: tweaks made by Michael Pritchard [pritcham]

2022, June: incorporating feature_freq_bands() added 2020, June by Solange Cerny [scerny], Aston University (https://github.com/fcampelo/EEG_Classification_)
"""

# Commented since not used. [fcampelo]
# import sys
#from scipy.spatial.distance import euclidean

import numpy as np
import scipy
import scipy.signal
import params
import pandas as pd
'''    import scipy.fftpack
import time    
import csv    
import os.path    
import scipy.io    
import json'''

def matrix_from_csv_file(file_path):
    """
    Returns the data matrix given the path of a CSV file.
    
    Parameters:
        file_path (str): path for the CSV file with a time stamp in the first column
            and the signals in the subsequent ones.
            Time stamps are in seconds, with millisecond precision

    Returns:
        numpy.ndarray: 2D matrix containing the data read from the CSV
    
    Author: 
        Original: [lmanso] 
        Revision and documentation: [fcampelo]
    
    """
    
    csv_data = np.genfromtxt(file_path, delimiter = ',')
    full_matrix = csv_data[1:]
    #headers = csv_data[0] # Commented since not used or returned [fcampelo]
    
    return full_matrix


def get_time_slice(full_matrix, start = 0., period = 1.):
    """
    Returns a slice of the given matrix, where start is the offset and period is 
    used to specify the length of the signal.
    
    Parameters:
        full_matrix (numpy.ndarray): matrix returned by matrix_from_csv()
        start (float): start point (in seconds after the beginning of records) 
        period (float): duration of the slice to be extracted (in seconds)

    Returns:
        numpy.ndarray: 2D matrix with the desired slice of the matrix
        float: actual length of the resulting time slice
        
    Author:
        Original: [lmanso]
        Reimplemented: [fcampelo]
    """
    
    # Changed for greater efficiency [fcampelo]
    rstart  = full_matrix[0, 0] + start
    index_0 = np.max(np.where(full_matrix[:, 0] <= rstart))
    index_1 = np.max(np.where(full_matrix[:, 0] <= rstart + period))
    
    duration = full_matrix[index_1, 0] - full_matrix[index_0, 0]
    return full_matrix[index_0:index_1, :], duration


def feature_raws(matrix):
    
    ret = matrix
    names = ['raw_' + str(i) for i in range(matrix.shape[1])]
    return ret, names




def calc_feature_vector(matrix, state):
    """
    Calculates all previously defined features and concatenates everything into 
    a single feature vector.
    
    Parameters:
        matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
        values of nsignals for a time window of length nsamples
        state (str): label associated with the time window represented in the 
        matrix.
        
    Returns:
        numpy.ndarray: 1D array containing all features
        list: list containing feature names for the features

    Author:
        Original: [lmanso]
        Updates and documentation: [fcampelo]
    """
    


    var_names = []    
    
    x, v = feature_raws(matrix)
    var_names += v
    var_values = x
    
    valsDF=pd.DataFrame(var_values,columns=var_names)
    flat=[valsDF[x].tolist() for x in valsDF.columns.values]
    valsFlat=pd.DataFrame([flat],columns=var_names)
    
    ''' THIS WORKS '''
    
    if state != None:
        valsFlat['Label'] = state
        var_names += ['Label']
   
    
   
   # if state != None:
   #     var_values = np.hstack([vals_squash.T, np.array([[state]],dtype=object)])
   #     var_names += ['Label']
        
    #var_values=var_values.real
    #forcing everything to be real because complex numbers can piss off
    return valsFlat, var_names



"""
Returns a number of feature vectors from a labeled CSV file, and a CSV header 
corresponding to the features generated.
full_file_path: The path of the file to be read
samples: size of the resampled vector
period: period of the time used to compute feature vectors
state: label for the feature vector
"""



def generate_raw_samples(file_path, nsamples, period, 
                                          state = None, 
                                          cols_to_ignore = None,
                                          output_file = None):
    """
    Reads data from CSV file in "file_path" and extracts statistical features 
    for each time window of width "period". 
    
    Details:
    Successive time windows overlap by period / 2. All signals are resampled to 
    "nsample" points to maintain consistency. 
    
    Parameters:
        file_path (str): file path to the CSV file containing the records
        nsamples (int): number of samples to use for each time window. The 
        signals are down/upsampled to nsamples
        period (float): desired width of the time windows, in seconds
        state(str/int/float): label to attribute to the feature vectors
         remove_redundant (bool): Should redundant features be removed from the 
        resulting feature vectors (redundant features are those that are 
        repeated due to the 1/2 period overlap between consecutive windows).
        cols_to_ignore (array): array of columns to ignore from the input matrix
         
        
    Returns:
        numpy.ndarray: 2D array containing features as columns and time windows 
        as rows.
        list: list containing the feature names

    Author:
        Original: [lmanso]
        Reimplemented: [fcampelo]
    """    
    # Read the matrix from file
    matrix = matrix_from_csv_file(file_path)
    
    pptID,_,rep=((file_path.split('/')[-1])[:-4]).split('-')
    
    if pptID[-2]=='_':
        #This accounts for data of the form 001_1, 001_2 rather than 001a, 001b
        pptID,run=map(int,pptID.split('_'))
    else:
        run=0 if pptID[-1].isdigit() else params.runletter_to_num[pptID[-1]]
        pptID=int(pptID) if pptID[-1].isdigit() else int(pptID[:-1])
    
    # We will start at the very begining of the file
    t = 0.
    
    # No previous vector is available at the start
    previous_vector = None
    
    # Initialise empty return object
    ret = None
    
    # Until an exception is raised or a stop condition is met
    while True:
        # Get the next slice from the file (starting at time 't', with a 
        # duration of 'period'
        # If an exception is raised or the slice is not as long as we expected, 
        # return the current data available
        try:
            s, dur = get_time_slice(matrix, start = t, period = period)
            if cols_to_ignore is not None:
                s = np.delete(s, cols_to_ignore, axis = 1)
        except IndexError:
            print('index error')
            break
        if len(s) == 0:
            print('slicelength 0')
            break
        if dur < 0.75 * period:
            print('slice duration '+str(dur)+' less than 75% of T '+str(period))
            break
        
        # Perform the resampling of the vector
        ry, rx = scipy.signal.resample(s[:, 1:], num = nsamples, 
                                 t = s[:, 0], axis = 0)
        
        # Slide the slice by 1/2 period
        t += 0.5 * period

        # Compute the feature vector. We will be appending the features of the 
        # current time slice and those of the previous one.
        # If there was no previous vector we just set it and continue 
        # with the next vector.
        r, headers = calc_feature_vector(ry, state)
        
        '''
        run=0 if isinstance(pptID,int) else params.runletter_to_num[pptID[-1]]
        pptID=int(pptID) if pptID[-1].isdigit() else int(pptID[:-1])
        
        if isinstance(pptID,int):
            run=0
        else:
            run=params.runletter_to_num[pptID[-1]]
            if pptID[-1].isdigit():
                pptID=int(pptID)
            else:
                pptID=int(pptID[:-1])    
        '''
        
        
        if previous_vector is not None:
            # If there is a previous vector, the script concatenates the two 
            # vectors and adds the result to the output matrix
            feature_vector = pd.concat([previous_vector.add_prefix('lag1_'), r],axis=1)
            id_vector=pd.DataFrame([[pptID,run,int(rep),t-0.5*period,t]],columns=['ID_pptID','ID_run','ID_gestrep','ID_tstart','ID_tend'])
            feature_vector=pd.concat([id_vector,feature_vector],axis=1)
            
            if ret is None:
                ret = feature_vector.copy()
            else:
                ret = pd.concat([ret, feature_vector],axis=0)
                
        # Store the vector of the previous window
        previous_vector = r.copy()
        if state is not None:
             # Remove the label (last column) of previous vector
            previous_vector.drop('Label',axis=1,inplace=True)

    feat_names = ['ID_pptID','ID_run','ID_gestrep','ID_tstart','ID_tend']+["lag1_" + s for s in headers[:-1]] + headers
    #catching unboundlocal error "headers before assignment" and skipping in the calling context of this function (ie in gen_train)
    

                
    '''if os.path.isfile(output_file):    
        with open(output_file, 'a', newline='') as data_file:    
            writer = csv.writer(data_file)    
            writer.writerow(feature_vector)    
    else:    
        with open(output_file, 'w', newline='') as data_file:    
            writer = csv.writer(data_file)    
            writer.writerow(feat_names)    
            writer.writerow(feature_vector)'''

    # Return
    return ret, feat_names


# ========================================================================
"""
Other notes by [fcampelo]:
1) ENTROPY
Entropy does not make sense for the "continuous" distribution of 
signal values. The closest analogue, Shannon's differential entropy, 
has been shown to be incorrect from a mathematical perspective
(see, https://www.crmarsh.com/static/pdf/Charles_Marsh_Continuous_Entropy.pdf
and https://en.wikipedia.org/wiki/Limiting_density_of_discrete_points )
I could not find an easy way to implement the LDDP here, nor any ready-to-use 
function, so I'm leaving entropy out of the features for now.
A possible alternative would be to calculate the entropy of a histogram of each 
signal. Also something to discuss.

2) CORRELATION
The way the correlations were calculated in the previous script didn't make 
much sense. What was being done was calculating the correlations of 75 pairs of 
vectors, each composed of a single observation of the 5 signals. I cannot think 
of any reason why this would be interesting, or carry any useful information
(simply because the first sample of h1 should be no more related to the first 
sample of h2 than it would be to the one immediately after - or before - it).
A (possibly) more useful information would be the correlations of each of the 
5 signals against each other (maybe they can be correlated under some mental 
states and decorrelated for others)? This is already done by the covariance 
matrix.

3) AUTOCORRELATION
A possibility would be to use the autocorrelation and cross-correlation of 
the signals. Both can be easily calculated, but would result in a massive 
amount of features (e.g., full autocorrelation would yield 2N-1 features per 
signal.). Not sure if we want that, but it's something to consider.

4) TSFRESH
Package tsfresh seemingly has a load of features implemented for time series,
it may be worth exploring.
"""
#
