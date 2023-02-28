#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:53:08 2020

@author: michael
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
        #Class=[]
        index=0
        headers=headers[:-1]
        headers.append('Class')
        emgwriter.writerow(tuple(list(headers)))
        for x in labels:
            datarow=dataset[index]
            writerow=list(datarow)
            toggle=1
            if (toggle==0):
                if x==0:
                    #Class.append('Open') # or sometimes Splay
                    """commenting out this unused approach.
                    will look again later as it might actually be better"""
                    writerow.append('Neutral')
                elif x==1:
                    #Class.append('Neutral')
                    writerow.append('Wave')
                elif x==2:
                    #Class.append('Close') # or sometimes Clench
                    writerow.append('Lshape')
                elif x==3:
                    #Class.append('Scissors') # or sometimes Peace
                    writerow.append('Point')
                elif x==4:
                    #Class.append('ThumbsUp')
                    writerow.append('Peace')
                elif x==5:
                    #Class.append('Paper') # or sometimes FlatOpen
                    writerow.append('Fist')
                elif x==6:
                    #Class.append('IdxPoint')
                    writerow.append('Wrist')
                elif x==7:
                    #Class.append('FlatClose')
                    writerow.append('Grip')
                else:
                    #Class.append('Undefined')
                    writerow.append('Undefined')
            else:
                if x==0:
                    writerow.append('Neutral')
                elif x==1:
                    writerow.append('0')
                elif x==2:
                    writerow.append('1')
                elif x==3:
                    writerow.append('2')
                elif x==4:
                    writerow.append('3')
                elif x==5:
                    writerow.append('4')
                elif x==6:
                    writerow.append('5')
                elif x==7:
                    writerow.append('6')
                elif x==8:
                    writerow.append('7')
                elif x==9:
                    writerow.append('8')
                elif x==10:
                    writerow.append('9')
                elif x==11:
                    writerow.append('a')
                elif x==12:
                    writerow.append('b')
                elif x==13:
                    writerow.append('c')
                elif x==14:
                    writerow.append('d')
                elif x==15:
                    writerow.append('e')
                elif x==16:
                    writerow.append('f')
                elif x==17:
                    writerow.append('g')
                elif x==18:
                    writerow.append('h')
                elif x==19:
                    writerow.append('i')
                elif x==20:
                    writerow.append('j')
                elif x==21:
                    writerow.append('k')
                elif x==22:
                    writerow.append('l')
                elif x==23:
                    writerow.append('m')
                elif x==24:
                    writerow.append('n')
                elif x==25:
                    writerow.append('o')
                elif x==26:
                    writerow.append('p')
                elif x==27:
                    writerow.append('q')
                elif x==28:
                    writerow.append('r')
                elif x==29:
                    writerow.append('s')
                elif x==30:
                    writerow.append('t')
                elif x==31:
                    writerow.append('u')
                elif x==32:
                    writerow.append('v')
                elif x==33:
                    writerow.append('w')
                elif x==34:
                    writerow.append('x')
                elif x==35:
                    writerow.append('y')
                elif x==36:
                    writerow.append('z')
                else:
                    writerow.append('Undefined')
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