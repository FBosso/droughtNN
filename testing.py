#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:58:23 2022

@author: francesco
"""

import unittest
#from LOO_model_selection_combo import generate_dataset
import pandas as pd
import tensorflow as tf
import os
import numpy as np


def generate_dataset(month,key):
    
    datasets = key.split('%')
    
    loc = []
    glob = []
    for item in datasets:
        if 'csv' in item:
            glob.append(item)
        else:
            loc.append(item)
     
    if month == '1':
        month = '12'
    else:
        month = str(int(month)-1)
    
    loc_data = []
    for item in loc:
        serie = timeseries_from_folder(item,month,1979,2021) #serie dovrà essere una lista
        loc_data.append(serie) #loc_data dovrà essere una lista di liste
        
    glob_data = []   
    for item in glob:
        glob_data.append(pd.read_csv(item, index_col=0))
    if len(glob_data) > 0:
        dataset = pd.concat(glob_data, axis=1)
        dataset.sort_values('year_glvar')
    else:
        dataset = pd.DataFrame()
        target_path = '/Users/francesco/Desktop/Università/Magistrale/Matricola/II_anno/Semestre_II/CLINT_project/Tesi/code/local_data/final_data/tp'
        if month == '12':
            month_target = '1'
        else:
            month_target = str(int(month)+1)
        dataset['target'] = timeseries_from_folder(target_path,month_target,1980,2021)
    
    for i,item in enumerate(loc_data):
        variable = loc[i].split('/')[-1]
        dataset[variable] = item
        
    return(dataset)



def timeseries_from_folder(path,month,startyr,endyr):
    
    files = os.listdir(path)
    files.sort()
    
    data = []
    for file in files:
        year = int(file.split('-')[0])
        if len(month) == 1:
            month = '0'+str(month)
        if ('-'+month in file) and (year >= startyr) and (year <= endyr):
            data.append(float(np.load(path+'/'+file)))
            if len(data) == (42):
                break
    return data



class TestTargetLoading(unittest.TestCase):
    
    def test_generateDataset1(self):
        dataset1 = pd.read_csv('/Users/francesco/Desktop/NeuralNetworks/testing_elements/dataset1.csv')
        path = '/Users/francesco/Desktop/Università/Magistrale/Matricola/II_anno/Semestre_II/CLINT_project/Tesi/code/local_data/final_data/t2m'
        self.assertEqual(generate_dataset('1',path)['t2m'],dataset1['t2m'])
        self.assetEqual(generate_dataset('1',path)['target'], dataset1['target'])
        
        
        
if __name__ == '__main__':
    unittest.main()