#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:14:40 2022

@author: francesco
"""


import pandas as pd
from functions_combo import generate_dataset, LOO_from_dataset
from skelm import ELMRegressor
import matplotlib.pyplot as plt
import numpy as np
import os


numbers = [i for i in range (2,13)]
activations = ['relu','sigm']
dict_results = {}

# loop for each month to search the best performin algorithm for each month
for i in range(1,13):
    # initialize list to store all the best values of each file for a specific month
    names = []
    values = []
    neurons = []
    activ = []
    # iterate over all the possible number of neurons
    for item in numbers:
        # initialize list to store all the best values for each activation functions fo a specific month
        names_temp = []
        values_temp = []
        neurons_temp = []
        activ_temp = []
        # iterate over the activation functions
        for activation in activations:
            # define the path of the file based on neurons and activation
            path = f'features_permutation_scores/skELM_neu-{item}_act-{activation}_scores.csv'
            # read the file
            a = pd.read_csv(path, index_col=0)
            # extract the name of the best performing model in that specific file for the considered month
            name = list(a[str(i)].loc[a[str(i)]==a[str(i)].min()].index)[0]
            # extract the MSE of the best performing model in that specific file for the considered month
            value = a[str(i)].loc[a[str(i)]==a[str(i)].min()][0]
            # save all the parameters in the temporar list to chose which activation functions overperform the others
            names_temp.append(name)
            values_temp.append(value)
            neurons_temp.append(item)
            activ_temp.append(activation)
        # identify the index in the list related to the best performing activation function
        index = values_temp.index(min(values_temp))
        # append all the data of the best performing model to the monthly list
        names.append(names_temp[index])
        values.append(values_temp[index])
        neurons.append(neurons_temp[index])
        activ.append(activ_temp[index])
    # identify the index of the best performin model in the monthly list 
    index = values.index(min(values))
    # store all the info of the model in the dictionary
    dict_results[i] = (names[index],values[index], neurons[index], activ[index])
    
months = list(dict_results.keys())
combos = []
neurons = []
activ = []    
for key in dict_results.keys():
    combos.append(dict_results[key][0])
    neurons.append(dict_results[key][2])
    activ.append(dict_results[key][3])
    
gen_strings = []   
for combo in combos:
    gen_string = ''
    variables = combo.split('-')
    for variable in variables:
        if '.csv' in variable:
            if variables[variables.index(variable)-3] == 'ENSO':
                ending_index = variables.index(variable)
                variables[ending_index-3:ending_index+1] = ['-'.join(variables[ending_index-3:ending_index+1])]
                
            else:
                ending_index = variables.index(variable)
                variables[ending_index-2:ending_index+1] = ['-'.join(variables[ending_index-2:ending_index+1])]
            
            
    for i,variable in enumerate(variables):
        if variable.split('.')[-1] == 'csv':
            base_path = 'data/global_data'
            
            ending_part = variable.split('-')[-1]
            folder = variable.replace('-'+ending_part,'')
            file = variable
            full_path = base_path+'/'+folder+'/'+file
            
            if i == 0:
                gen_string = full_path
            elif i != 0:
                gen_string = gen_string+'%'+full_path
            
        else:
            base_path = 'data/local_data'
            folder = variable
            
            full_path = base_path+'/'+variable
            
            if i == 0:
                gen_string = full_path
            elif i != 0:
                gen_string = gen_string+'%'+full_path
    
    gen_strings.append(gen_string)


tot_hat_elm = []
tot_true = []
tot_years_elm = []
for month,combo,neuron,activation in zip(months,gen_strings,neurons,activ):
    #create dataset
    dataset = generate_dataset(str(month),combo)
    #define ELM hyperparams
    hyperparams = {
        'neuron': neuron,
        'activation': activation
        }
    #save LOO points for linear model
    true, predicted, years = LOO_from_dataset(combo,dataset,'skELM',hyperparams, save_points=True, already_normalized=False, yr=True)

    #lists with true values, predicted values, and relate years
    tot_true.append(true)
    tot_hat_elm.append(predicted)
    tot_years_elm.append(years)
    
base_path = 'data/ECMWF_benchmark'
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
    plt.savefig(f'/Users/francesco/Desktop/ELM_ECMWF/{month}_ELM-ECMWF.pdf', bbox_inches='tight')
    plt.close()
    
    
    
    
    
        

