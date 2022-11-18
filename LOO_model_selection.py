#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 12:49:57 2022

@author: francesco
"""

#import section
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from functions import loo_from_lists
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#tf.keras.utils.disable_interactive_logging()


###identify and load the data of valid combinations per each months
folders_path = '/Users/francesco/Desktop/newNIPA/output'

#loading all the dataset paths
files = []
folders = os.listdir(folders_path)
try:
    folders.remove('.DS_Store')
except:
    pass

for folder in folders:  
    for file in os.listdir(folders_path+'/'+folder):
        if (file.split('_')[-1] == 'dataset.csv') and (file.split('_')[-2].split('-')[0] == 'tp') :
            files.append(folders_path+'/'+folder+'/'+file)
            

            
            
#dividing datasets by moths
months = [i+1 for i in range(12)]
months_dict = {
     '1':[],
     '2':[],
     '3':[],
     '4':[],
     '5':[],
     '6':[],
     '7':[],
     '8':[],
     '9':[],
     '10':[],
     '11':[],
     '12':[]
    }

for item in files:
    for month in months:
        if item.split('_')[-2].split('-')[1] == str(month):
            months_dict[str(month)].append(item)


#initialize variables to store the results
resultsELM = {
    '1':{},
    '2':{},
    '3':{},
    '4':{},
    '5':{},
    '6':{},
    '7':{},
    '8':{},
    '9':{},
    '10':{},
    '11':{},
    '12':{}
    }

resultsNN = {
    '1':{},
    '2':{},
    '3':{},
    '4':{},
    '5':{},
    '6':{},
    '7':{},
    '8':{},
    '9':{},
    '10':{},
    '11':{},
    '12':{}
    }

###build N models for each month with each one of the N founded valid features 
###and compute the LOO validation error
models = ['ELM','NN']
for model in models:
    for month in months_dict:
        print(f'Month {month} Running ... ')
        LOO = loo_from_lists(months_dict[month], model)
        print(f'Month {month} DONE\n')
        only_names = []
        for item in months_dict[month]:
            only_names.append(item.split('/')[-1].split('.')[0])
        res = dict(zip(only_names, LOO))
        if model == 'ELM':
            resultsELM[month] = res
        if model == 'NN':
            resultsNN[month] = res

ELM = pd.DataFrame(resultsELM)
ELM.to_csv('features_permutation_scores/ELM_scores.csv')

NN = pd.DataFrame(resultsNN)
NN.to_csv('features_permutation_scores/NN_scores.csv')
    


for key in resultsELM.keys():
    label = list(resultsELM[key].keys())
    valuesELM = np.log(np.array(list(resultsELM[key].values())))
    valuesNN = np.log(np.array(list(resultsNN[key].values())))
    plt.barh(label,valuesELM,color='g', alpha=0.5, label='ELM')
    plt.barh(label,valuesNN,color='b', alpha=0.5,label='NN')
    plt.legend()
    
    plt.show()


df = pd.DataFrame(resultsNN)

bestNN = []
for item in list(df.columns):
    month = item
    value = df[item].min()
    name = df.loc[df[item] == df[item].min()]
    name = list(name.index)
    a = (month, name, value)
    bestNN.append(a)


df = pd.DataFrame(resultsELM)

bestELM = []
for item in list(df.columns):
    month = item
    value = df[item].min()
    name = df.loc[df[item] == df[item].min()]
    name = list(name.index)
    a = (month, name, value)
    bestELM.append(a)
    














