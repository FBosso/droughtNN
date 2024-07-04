#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:02:47 2023

@author: francesco
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:14:40 2022

@author: francesco
"""


import pandas as pd
from functions_combo import generate_dataset, LOO_from_dataset, list2name_NIPA, best_model_from_csvs
from skelm import ELMRegressor
import matplotlib.pyplot as plt
import numpy as np
import os


neurons = [i for i in range (6,13)]
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
    true_years = np.array(years[start:end+1])
    
    
    
    #June (5 because index starts from 0)
    if i == 5:
        np.save('../../tot_trend_comparison/June/ELM',hat_elm_partial)
    #July (6 because index starts from 0)
    elif i == 6:
        np.save('../../tot_trend_comparison/July/ELM',hat_elm_partial)
    
    
    
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
    plt.savefig(f'plots/ELM_ECMWF/{month}_ELM-ECMWF.pdf', bbox_inches='tight')
    plt.close()
    
    #SHIFT CHECK######
    #ECMWF = np.delete(ECMWF,-1)
    #ECMWF = np.append(ECMWF,0)
    #ECMWF = np.insert(ECMWF,0,0)
    ######## 
    if (month == '06') or (month == '07'):
        
        month_dict = {
            6:'June',
            7:'July'
            }
        
        true_ECMWF = np.load(f'plots/true_ECMWF_targets/real_ECMWF_target_{int(month)}.npy')
        
        plt.figure(figsize=(12,6))
    
        plt.ylim(0,170)
        
        plt.xlabel(f'Year (month={int(month)})')
        plt.ylabel('Cumulative precipitation [mm]')
        
        plt.xticks(true_years)
        '''
        from scipy.interpolate import make_interp_spline
        
        X_Y_Spline_true_ECMWF = make_interp_spline(true_years,true_ECMWF)
        X_Y_Spline_true_elm = make_interp_spline(true_years,true_partial)
        X_Y_Spline_elm = make_interp_spline(true_years,hat_elm_partial)
        X_Y_Spline_ECMWF = make_interp_spline(true_years,ECMWF)
        
        
        X_true_elm = np.linspace(true_years.min(), true_years.max(), 20)
        Y_true_elm = X_Y_Spline_true_elm(X_true_elm)
        
        X_elm = np.linspace(true_years.min(), true_years.max(), 20)
        Y_elm = X_Y_Spline_elm(X_elm)
        
        
        X_true_ECMWF = np.linspace(true_years.min(), true_years.max(), 20)
        Y_true_ECMWF = X_Y_Spline_ECMWF(X_true_ECMWF)
        
        X_ECMWF = np.linspace(true_years.min(), true_years.max(), 20)
        Y_ECMWF = X_Y_Spline_true_ECMWF(X_ECMWF)
        
        plt.plot(X_true_elm,Y_true_elm, label='ELM target (Observation)', c='black', marker='*')
        plt.plot(X_elm,Y_elm, label='ELM Prediction', c='b')
        plt.fill_between(X_true_elm,Y_elm,Y_true_elm, alpha=0.3, color='yellow')
        '''
        plt.plot(true_years,true_partial, label='ELM target (Observation)', c='black', marker='*')
        plt.plot(true_years,hat_elm_partial, label='ELM Prediction', c='b')
        plt.fill_between(true_years,true_partial,hat_elm_partial, alpha=0.3, color='yellow')
        
        '''
        plt.plot(X_true_ECMWF,Y_ECMWF, label='ECMWF Predictions')
        plt.plot(X_true_ECMWF,Y_true_ECMWF, label='ECMWF Target')
        plt.fill_between(X_true_ECMWF,Y_ECMWF,Y_true_ECMWF, alpha=0.5, color='purple')
        '''
        plt.legend(loc="upper left")
        plt.savefig(f'plots/{month}_trueVSelm.pdf')
        plt.show()
        
        
        
        
        
        
        
        plt.figure(figsize=(12,6))
    
        plt.ylim(0,170)
        
        plt.xlabel(f'Year (month={int(month)})')
        plt.ylabel('Cumulative precipitation [mm]')
        
        plt.xticks(true_years)
        '''
        from scipy.interpolate import make_interp_spline
        
        X_Y_Spline_true_ECMWF = make_interp_spline(true_years,true_ECMWF)
        X_Y_Spline_true_elm = make_interp_spline(true_years,true_partial)
        X_Y_Spline_elm = make_interp_spline(true_years,hat_elm_partial)
        X_Y_Spline_ECMWF = make_interp_spline(true_years,ECMWF)
        
        
        X_true_elm = np.linspace(true_years.min(), true_years.max(), 20)
        Y_true_elm = X_Y_Spline_true_elm(X_true_elm)
        
        X_elm = np.linspace(true_years.min(), true_years.max(), 20)
        Y_elm = X_Y_Spline_elm(X_elm)
        
        
        X_true_ECMWF = np.linspace(true_years.min(), true_years.max(), 20)
        Y_true_ECMWF = X_Y_Spline_ECMWF(X_true_ECMWF)
        
        X_ECMWF = np.linspace(true_years.min(), true_years.max(), 20)
        Y_ECMWF = X_Y_Spline_true_ECMWF(X_ECMWF)
        
        plt.plot(X_true_elm,Y_true_elm, label='Obserations')
        plt.plot(X_elm,Y_elm, label='ELM Predictions')
        plt.fill_between(X_true_elm,Y_elm,Y_true_elm, alpha=0.5, color='orange')
        '''

        plt.plot(true_years,true_ECMWF, label='ECMWF target (Observation)', c='black', marker='*')
        plt.plot(true_years, ECMWF, label='ECMWF Prediction', c='purple')
        plt.fill_between(true_years,ECMWF,true_ECMWF, alpha=0.3, color='yellow')
        
        plt.legend(loc="upper left")
        plt.savefig(f'plots/{month}_trueVSecmwf.pdf')
        plt.show()
        
        
        
        ########## all together ##########
        
        true_ECMWF = np.load(f'plots/true_ECMWF_targets/real_ECMWF_target_{int(month)}.npy')
        
        plt.figure(figsize=(12,6))
    
        plt.ylim(0,170)
        
        plt.xlabel(f'Year (month={month_dict[int(month)]})')
        plt.ylabel('Cumulative precipitation [mm]')
        
        plt.xticks(true_years)
        
        plt.plot(true_years,true_partial, label='Observation', c='black', marker='*')
        plt.plot(true_years,hat_elm_partial, label='Monthly ML Prediction', c='b')
        plt.plot(true_years, ECMWF, label='ECMWF Prediction', c='purple')
        
        plt.legend(loc="upper left")
        plt.ylim(0,180)
        plt.savefig(f'plots/{month}_trueVSelmVSecmwf.pdf')
        plt.show()
        






    

