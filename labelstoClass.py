#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:53:08 2020

@author: pritcham
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import csv

def numtoClass(input_file, output_file):
    with open((output_file),'w',newline='') as csvfile:
        emgwriter=csv.writer(csvfile, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_data = np.genfromtxt(input_file, delimiter=',')
        dataset = csv_data[1:,0:-1]
        labels = csv_data[1:,-1]
        with open(input_file) as readheads:
            headers=csv.DictReader(readheads).fieldnames
        #headers = csv_data.dtype.names
        Class=[]
        index=0
        headers=headers[:-1]
        headers.append('Class')
        emgwriter.writerow(tuple(list(headers)))
        for x in labels:
            datarow=dataset[index]
            writerow=list(datarow)
            if x==0:
                Class.append('Open') #Open
                writerow.append('Open')
            elif x==1:
                Class.append('Neutral')
                writerow.append('Neutral')
            elif x==2:
                Class.append('Close') #Close
                writerow.append('Close')
            else:
                Class.append('Undefined')
            writerow=tuple(writerow)
            emgwriter.writerow(writerow)
            index=index+1
        print('conversion success')
        #emgwrite=list(emg)
        #emgwrite.append(int(round(time.time() * 1000)))
        #emgwrite=tuple(emgwrite)
        #emgwriter.writerow(emgwrite)


if __name__ == '__main__':
	if len(sys.argv) <3:
		print ('arg1: input dir\narg2: output file')
		sys.exit(-1)
	input_file = sys.argv[1]
	output_file = sys.argv[2]
	numtoClass(input_file, output_file)