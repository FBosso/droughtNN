#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:52:44 2022

@author: francesco
"""

#import section
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import itertools
from functions_combo import pretty_combo, tuple2key, generate_dataset, LOO_from_dataset
from tqdm import tqdm

#tf.keras.utils.disable_interactive_logging()


###identify and load the data of valid combinations per each months
folders_path = '../data/global_data'

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
            
# definition of local variables

base_path = 'data/local_data/'
local_variables_names = ['MER','MSSHF','RH','SD','SH','t2m','TCC','TCWV','tp','UW','VW']
#local_variables_names = ['SD']

local_variables = [base_path+var for var in local_variables_names]

print('Start Combo creation ... \n')
#creation of a dictionary with keys reporting all the possibe monthly combo 
#of global and local variables
for month in months_dict.keys():
    if len(months_dict[month]) > 0:
        variables = local_variables.copy()
        
        for dataset in months_dict[month]:
            variables.append(dataset) # testare
        combo = []
        for i in range(len(variables)):
            for combination in itertools.combinations(variables,i+1):
                # take into account only combination with maximum 4 elemens
                if len(combination) <= 4:
                    # consider only the combinations with maximum 1 global signal
                    c = 0
                    for element in combination:
                        if 'csv' in element:
                            c += 1
                    if c <= 2:
                        combo.append(combination)
                        
        keys = []
        for item in combo:
            keys.append(tuple2key(item))
        values = [0 for i in range(len(keys))]
        diz = dict(zip(keys, values))
        months_dict[month] = diz
            
    else:
        variables = local_variables.copy()
        combo = []
        for i in range(len(variables)):
            for combination in itertools.combinations(variables,i+1):
                if len(combination) > 0 and len(combination)<=4:
                    combo.append(combination)
        keys = []
        for item in combo:
            keys.append(tuple2key(item))
        values = [0 for i in range(len(keys))]
        diz = dict(zip(keys,values))
        months_dict[month] = diz
        
        
    print(f'\tCombo month {month} CREATED')
    
print('\nCombo Creation DONE\n')
    
models = ['skELM']
for model in models:
    if (model == 'ELM') or (model == 'skELM'):
        # definition of hyperparameter subsets to search for the best model
        neurons = [6,7,8,9,10,11,12]
        #neurons = [2,3,4,5,6,7,8,9,10,11,12]
        activations = ['relu','sigm']
        for neuron in tqdm(neurons, desc='ELM creation',leave=True):
            for activation in activations:
                #print(f'\nMODEL:{model}\tHYPERPARAMS:neu-{neuron}\n')
                for month in months_dict.keys():
                    hyperparams = {
                        'neuron':neuron,
                        'activation': activation
                        }
                    
                    #print(f'\nStart month {month} ...')
                    for combo in months_dict[month].keys():
                        dataset = generate_dataset(month,combo)
                        LOO = LOO_from_dataset(combo,dataset,model,hyperparams)
                        months_dict[month][combo] = LOO
                    #print(f'Month {month} \tdone\n ')
                
                df = pd.DataFrame(months_dict)
                new_index = []
                for item in df.index:
                    new_index.append(pretty_combo(item))
                df = df.set_index(pd.Index(new_index))
                hyper_setting = f'neu-{neuron}_act-{activation}'
                df.to_csv(f'/Users/francesco/Desktop/NeuralNetworks/features_permutation_scores/{model}_{hyper_setting}_scores.csv')
        
    
    elif (model == 'NN') or (model == 'torchNN'):
        #definition of hyperparameter subsets to search for the best model
        neurons = [4,5,6,7]
        #batch_sizes = [1,2,3,4,5]
        batch_sizes = [1,2,3,4,5,6]
        learning_rates = [0.01,0.05,0.1]
        activations = ['relu','selu','elu']
        #epochs = [20,25,30,35,40,45,50,55,60,65,70]
        epochs = [30,35,40,45,50]
        


        for neuron in tqdm(neurons, desc='\nNN cration', leave=True):
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    for activation in activations:
                        for epoch in epochs:
                            #print(f'\nMODEL:{model}\tHYPERPARAMS:neu-{neuron}_bs-{batch_size}_lr-{learning_rate}_act-{activation}_ep-{epoch}\n')
                            for month in tqdm(months_dict.keys(), desc=f'NN Creation: neu-{neuron}_bs-{batch_size}_lr-{learning_rate}_act-{activation}_ep-{epoch}', leave=True):
                                
                                hyperparams = {
                                    'neuron':neuron,
                                    'batch_size':batch_size,
                                    'learning_rate':learning_rate,
                                    'activation':activation,
                                    'epoch':epoch
                                    }
                                
                                #print(f'\nStart month {month} ...')
                                for combo in months_dict[month].keys():
                                    dataset = generate_dataset(month,combo)
                                    LOO = LOO_from_dataset(combo,dataset,model,hyperparams)
                                    months_dict[month][combo] = LOO
                                #print(f'Month {month} \tdone\n ')
    
                            df = pd.DataFrame(months_dict)
                            new_index = []
                            for item in df.index:
                                new_index.append(pretty_combo(item))
                            df = df.set_index(pd.Index(new_index))
                            hyper_setting = f'neu-{neuron}_bs-{batch_size}_lr-{learning_rate}_act-{activation}_ep-{epoch}'
                            df.to_csv(f'/Users/francesco/Desktop/NeuralNetworks/features_permutation_scores/{model}_{hyper_setting}_scores.csv')

    
    

        
            
            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            