#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import sys
sys.path.insert(0, '../ML_algos/dataset/')

def read_csv(file):
    outfile=[]
    with open(file) as csvfile:
        csvread = csv.reader(csvfile,delimiter=",")
        for row in csvread:
                outfile.append(row)            
        return outfile

        