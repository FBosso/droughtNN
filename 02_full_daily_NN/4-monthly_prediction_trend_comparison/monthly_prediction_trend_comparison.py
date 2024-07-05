#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:40:38 2023

@author: francesco
"""
#import section
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from function_full import params_based_normalization, save_trend_comparison
import matplotlib.pyplot as plt


#needed paths
models_path = '../2-hyperparameters_tuning/tuner_trials/best_hyperparams'
dataset_name = 'MSSHF-SH-t2m-TCC-tp-UW-VW-MSLP-Z500'
testing_data_path = '../3-best_models/best_data'
base_path = '../../data/ECMWF_benchmark'


#load model
model = tf.keras.models.load_model(f'{models_path}/{dataset_name}/model')

#load training data
x_train = pd.read_csv(f'{testing_data_path}/{dataset_name}/x_train_{dataset_name}', index_col = 0)
y_train = pd.read_csv(f'{testing_data_path}/{dataset_name}/y_train_{dataset_name}', index_col = 0)
train = pd.concat([x_train,y_train], axis=1)
train['label'] = 'train'

#load testing data
x_test = pd.read_csv(f'{testing_data_path}/{dataset_name}/x_test_{dataset_name}', index_col = 0)
y_test = pd.read_csv(f'{testing_data_path}/{dataset_name}/y_test_{dataset_name}', index_col = 0)
test = pd.concat([x_test,y_test], axis=1)
test['label'] = 'test'

#load normalization parameters
means = np.load(f'{testing_data_path}/{dataset_name}/means_{dataset_name}.npy')
stds = np.load(f'{testing_data_path}/{dataset_name}/stds_{dataset_name}.npy')

tot = pd.concat([train,test], axis=0)
tot = tot.sort_index()

base_start_year = 1979
base_end_year = 2021

files = os.listdir(base_path)
files.sort()


#month_day = {1:[1,2],2:[1,30],3:[3,2],4:[4,2],5:[4,30],6:[6,1],7:[7,1],8:[8,1],9:[9,2],10:[9,30],11:[10,31],12:[12,2]}

month_day = {1:[12,3],2:[12,31],3:[1,31],4:[3,3],5:[3,31],6:[5,2],7:[6,1],8:[7,2],9:[8,3],10:[8,31],11:[10,1],12:[11,2]}


ECMWF_MSEs = []
for i,file in enumerate(files):
    #load the file
    ECMWF = np.load(base_path+'/'+file)
    #detect the information in the file's name
    month = int(file.split('_')[0])
    startyr = int(file.split('_')[-1].split('.')[0].split('-')[0])
    endyr = int(file.split('_')[-1].split('.')[0].split('-')[1])
    true_years = np.array([i for i in range(startyr,endyr+1)])
    #detect the list of true values, predicted values, and years for the specific month
    
    i = i+1
    
    if i == 1 or i == 2 or i == 12:
        base_end_year = 2020
        base_start = startyr - base_start_year
        base_end = base_end_year - endyr
        extremes = tot.loc[(tot['beginning_day'] == month_day[i][1]) & (tot['beginning_month'] == month_day[i][0])]
        
        start = base_start
        end = len(extremes)-base_end
        
        extremes = extremes.iloc[start:end]
        #extremes = extremes.tail(len(extremes)-base_start)
        #extremes = extremes.head(len(extremes)-base_end)
    
   
    
    else:
        base_start = startyr - base_start_year
        base_end = base_end_year - endyr + 1
        extremes = tot.loc[(tot['beginning_day'] == month_day[i][1]) & (tot['beginning_month'] == month_day[i][0])]
        
        start = base_start
        end = len(extremes)-base_end
        
        extremes = extremes.iloc[start:end]
        #extremes = extremes.tail(len(extremes)-base_start)
        #extremes = extremes.head(len(extremes)-base_end)
        
    
    #df = extremes[['MSSHF', 'SH', 't2m', 'TCC', 'tp','UW', 'VW', 'msl', 'z']] 
    #extremes[['MSSHF', 'SH', 't2m', 'TCC', 'tp','UW', 'VW', 'msl', 'z']]  = (df - means)/stds
    
    samples = extremes[['MSSHF', 'SH', 't2m', 'TCC', 'tp','UW', 'VW', 'msl', 'z']].to_numpy()
    samples_norm = params_based_normalization(samples, means, stds)
    
    true = extremes['target'].to_numpy()
    
    MSE = ((true - ECMWF)**2).mean()
    ECMWF_MSEs.append(MSE)
    
    
    hat = model.predict(samples_norm)
    
    #save the data to produce the plot of each month into a prefixed folder (check function to know more)
    save_trend_comparison(i,hat,true,ECMWF,true_years)
    
    
    ### Plot data and save real ECMWF targets    
    #month dict to translate numbers into names
    month_dict = {
        1:'January',
        2:'February',
        3:'March',
        4:'April',
        5:'May',
        6:'June',
        7:'July',
        8:'August',
        9:'September',
        10:'October',
        11:'November',
        12:'December'
        }
   
    np.save(f'targets/real_ECMWF_target_{i}', true)
 
    plt.figure(figsize=(12,6))
     
    plt.ylim(0,180)
     
    plt.xlabel(f'Year (month={month_dict[i]})')
    plt.ylabel('Cumulative precipitation [mm]')
     
    plt.xticks(true_years)
     
    from scipy.interpolate import make_interp_spline
     
    X_Y_Spline_true = make_interp_spline(true_years,true)
    X_Y_Spline_elm = make_interp_spline(true_years,hat)
    X_Y_Spline_ECMWF = make_interp_spline(true_years,ECMWF)
     
    X_true = np.linspace(true_years.min(), true_years.max(), 20)
    Y_true = X_Y_Spline_true(X_true)
     
    X_elm = np.linspace(true_years.min(), true_years.max(), 20)
    Y_elm = X_Y_Spline_elm(X_elm)
     
    X_ECMWF = np.linspace(true_years.min(), true_years.max(), 20)
    Y_ECMWF = X_Y_Spline_ECMWF(X_ECMWF)
     
    plt.plot(X_true,Y_true, label='Obseration', c='black', marker='*')
    plt.plot(X_elm,Y_elm, label='Yearly ML Prediction', c='green')
    plt.plot(X_ECMWF,Y_ECMWF, label='ECMWF Prediction', c='purple')
    plt.legend(loc="upper left")
    plt.ylim(0,180)
    plt.savefig(f'plots/{i}_trueVSelmVSecmwf.pdf')
    plt.show()
            
        