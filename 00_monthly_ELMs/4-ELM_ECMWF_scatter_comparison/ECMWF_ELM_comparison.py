#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:14:40 2022

@author: francesco
"""


import pandas as pd
from functions_combo import generate_dataset, LOO_from_dataset, best_model_from_csvs
from skelm import ELMRegressor
import matplotlib.pyplot as plt
import numpy as np
import os


neurons = [i for i in range (4,13)]
activations = ['relu','sigm']


############### BEST MODEL SECECTION ###############
dict_results = best_model_from_csvs(neurons,activations)
    
tot_true = []
tot_hat_elm = []
tot_years_elm = []

for key in dict_results.keys():
    tot_true.append(dict_results[key][-1][0])
    tot_hat_elm.append(dict_results[key][-1][1])
    tot_years_elm.append(dict_results[key][-1][2])
    
base_path = '../../data/ECMWF_benchmark'
files = os.listdir(base_path)
files.sort()

for i,file in enumerate(files):
    #load the file
    ECMWF = np.load(base_path+'/'+file)
    #detect the information in the file's name
    month = file.split('_')[0]
    startyr = int(file.split('_')[-1].split('.')[0].split('-')[0])
    endyr = int(file.split('_')[-1].split('.')[0].split('-')[1])
    #delect the list of true values, predicted values, and years for the specific month
    true = tot_true[i]
    hat_elm = tot_hat_elm[i]
    years = tot_years_elm[i]
    #find the index of the start and end years of ECMWF inside the years of the ELM
    start = years.index(startyr)
    end = years.index(endyr)
    #select true and predicted data based on the index interval just founded
    true_partial = true[start:end+1]
    hat_elm_partial = hat_elm[start:end+1]
    
    #computation comarison data
    pearson_ELM = round(np.corrcoef(true_partial,hat_elm_partial)[0][1],2)
    pearson_ECMWF = round(np.corrcoef(true_partial,ECMWF)[0][1],2)
    MSE_ELM = round(((np.array(true_partial)-np.array(hat_elm_partial))**2).mean(),2)
    MSE_ECMWF = round(((np.array(true_partial)-np.array(ECMWF))**2).mean(),2)
    
    plt.scatter(true_partial,hat_elm_partial, label='ELM')
    plt.scatter(true_partial,ECMWF, label='ECMWF')
    plt.legend()
    plt.title(f'Month: {month}\n Pearson_ELM: {pearson_ELM} | Pearson_ECMWF: {pearson_ECMWF}\n MSE_ELM: {MSE_ELM} | MSE_ECMWF: {MSE_ECMWF}')
    plt.xlim(0,140)
    plt.ylim(0,140)
    #plt.show()
    plt.savefig(f'ELM_ECMWF/{month}_ELM-ECMWF.pdf', bbox_inches='tight')
    plt.close()
    

    
    
    
        

