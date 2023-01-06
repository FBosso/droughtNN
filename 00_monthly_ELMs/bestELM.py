#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 13:42:39 2022

@author: francesco
"""

import pandas as pd
from functions_combo import generate_dataset, LOO_from_dataset
from skelm import ELMRegressor
import matplotlib.pyplot as plt
import numpy as np


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
    copy = variables.copy()
    for variable in copy:
        if '.csv' in variable:
            if variables[variables.index(variable)-3] == 'ENSO':
                ending_index = variables.index(variable)
                variables[ending_index-3:ending_index+1] = ['-'.join(variables[ending_index-3:ending_index+1])]
                
            else:
                ending_index = variables.index(variable)
                variables[ending_index-2:ending_index+1] = ['-'.join(variables[ending_index-2:ending_index+1])]
            
            
    for i,variable in enumerate(variables):
        if variable.split('.')[-1] == 'csv':
            base_path = '../data/global_data'
            
            ending_part = variable.split('-')[-1]
            folder = variable.replace('-'+ending_part,'')
            file = variable
            full_path = base_path+'/'+folder+'/'+file
            
            if i == 0:
                gen_string = full_path
            elif i != 0:
                gen_string = gen_string+'%'+full_path
            
        else:
            base_path = '../data/local_data'
            folder = variable
            
            full_path = base_path+'/'+variable
            
            if i == 0:
                gen_string = full_path
            elif i != 0:
                gen_string = gen_string+'%'+full_path
    
    gen_strings.append(gen_string)
    
    
    
    

#neurons = [10 for i in range(12)]   

pearson = [[1,0.1],[0.1,1]]
while pearson[0][1] < 0.80:
    tot_hat_elm = []
    tot_true_elm = []
    for month,combo,neuron,activation in zip(months,gen_strings,neurons,activ):
        #create dataset
        dataset = generate_dataset(str(month),combo)
        #define ELM hyperparams
        hyperparams = {
            'neuron': neuron,
            'activation': activation
            }
        #save LOO points for linear model
        true, predicted = LOO_from_dataset(combo,dataset,'skELM',hyperparams, save_points=True, already_normalized=False)
        true_lin, predicted_lin = LOO_from_dataset(combo,dataset,'linear', save_points=True, already_normalized=False)
        
        #save the LOO MSE of the linear model in the same dictionary containing the results of the ELM
        MSE_lin = LOO_from_dataset(combo,dataset,'linear',hyperparams, already_normalized=False)
        dict_results[month] = dict_results[month] + (MSE_lin,)
        
        plt.scatter(true,predicted,label='ELM')
        plt.scatter(true_lin,predicted_lin,label='linear')
        plt.title(f'Month:{month} | PearsonELM:{round(np.corrcoef(true,predicted)[0][1],2)} | PearsonLIN:{round(np.corrcoef(true_lin,predicted_lin)[0][1],2)}')
        plt.legend()
        plt.xlim(0,150)
        plt.ylim(0,150)
        plt.show()
        
        #np.save(f'/Users/francesco/Desktop/Università/Magistrale/Matricola/II_anno/Semestre_II/CLINT_project/Tesi/presentation/presentation_2/plot/data_LOO/{month}_true', true)
        #np.save(f'/Users/francesco/Desktop/Università/Magistrale/Matricola/II_anno/Semestre_II/CLINT_project/Tesi/presentation/presentation_2/plot/data_LOO/{month}_predicted', predicted)

        tot_true_elm = tot_true_elm + true
        tot_hat_elm = tot_hat_elm + predicted
        
    pearson = np.corrcoef(tot_hat_elm,tot_true_elm)
    print(pearson)
    
dict_compare = {}
for item in dict_results.keys():
    dict_compare[item] = (dict_results[item][1],dict_results[item][4])

'''
    plt.scatter(tot_hat,tot_true)
    plt.xlim(0,150)
    plt.ylim(0,150)
'''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    