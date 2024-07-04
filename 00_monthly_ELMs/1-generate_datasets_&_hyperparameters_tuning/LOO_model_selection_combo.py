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

################ GLOBAL DATA ##################
###identify and load the data of valid combinations per each months
folders_path = '../../data/global_data'
#loading all the dataset paths
files = []
folders = os.listdir(folders_path)
try:
    folders.remove('.DS_Store')
except:
    pass
#append all the files inside a specific folder to the files list
for folder in folders:  
    for file in os.listdir(folders_path+'/'+folder):
        if (file.split('_')[-1] == 'dataset.csv') and (file.split('_')[-2].split('-')[0] == 'tp') :
            files.append(folders_path+'/'+folder+'/'+file)          
#dividing datasets by moths
months = [i+1 for i in range(12)]
months_dict = {
     '1':[],'2':[],'3':[],'4':[],'5':[],'6':[],
     '7':[],'8':[],'9':[],'10':[],'11':[],'12':[]
    }
#store the filename according to the month of the dict
for item in files:
    for month in months:
        if item.split('_')[-2].split('-')[1] == str(month):
            months_dict[str(month)].append(item)
            
            
################ LOCAL DATA ##################            
# definition of local variables
base_path = '../../data/local_data/'
local_variables_names = ['MER','MSSHF','RH','SD','SH','t2m','TCC','TCWV','tp','UW','VW']
#concatenation of path and vars name
local_variables = [base_path+var for var in local_variables_names]


################ COMBO CREATION ################## 
print('Start Combo creation ... \n')
#creation of a dictionary with keys reporting all the possibe monthly combo of global and local variables
for month in months_dict.keys():
    #check if the the considered month has global data coming fron NIPA
    if len(months_dict[month]) > 0:
        #store the local vars in a list
        variables = local_variables.copy()
        #append global variables to the same list of local ones
        for dataset in months_dict[month]:
            variables.append(dataset)
        #initialize list to store combinations
        combo = []
        #generate combinations and store them in the combo list
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
            
            #transform tuple to concatenation of variable path with separating %
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
        #initialize a zero vector to be replaced with accuracy values later on
        values = [0 for i in range(len(keys))]
        #create a dictionary matching every value to every combination of variable
        diz = dict(zip(keys,values))
        #assign this data structure to the month of the current iteration
        months_dict[month] = diz
        
        
    print(f'\tCombo month {month} CREATED')
print('\nCombo Creation DONE\n')


################ MODEL TRAINING ################## 
#list of all possible ELM options (maybe differemnt libraries)
models = ['skELM']
#for each ELM option
for model in models:
    # definition of hyperparameter subsets to search for the best model
    neurons = [6,7,8,9,10,11,12]
    activations = ['relu','sigm']
    #iterate over all hyperparameters combinations
    for neuron in tqdm(neurons, desc='ELM creation',leave=True):
        for activation in activations:
            #print(f'\nMODEL:{model}\tHYPERPARAMS:neu-{neuron}\n')
            #iterate across each month
            for month in months_dict.keys():
                #fix the hyperparams for the current month
                hyperparams = {
                    'neuron':neuron,
                    'activation': activation
                    }
                #print(f'\nStart month {month} ...')
                #iterate across the combos (with the hyperparameters previously fixed)
                for combo in months_dict[month].keys():
                    #generate the dataset by passing the month and the combo of variables we want to generate
                    dataset = generate_dataset(month,combo)
                    #perform training and compute Leave One Out error
                    LOO = LOO_from_dataset(combo,dataset,model,hyperparams)
                    #store the LOO error obtained in the dict
                    months_dict[month][combo] = LOO
                #print(f'Month {month} \tdone\n ')
            
            #after the iteration over all months is done, transform dict into dataframe
            df = pd.DataFrame(months_dict)
            #declare the list to store the new index
            new_index = []
            for item in df.index:
                #make the index of the dataframe just a bit prettier to read
                new_index.append(pretty_combo(item))
            #set the pretty index
            df = df.set_index(pd.Index(new_index))
            #create the name of the file storing all the LOO results for all the months in a specific hyperparams setting
            hyper_setting = f'neu-{neuron}_act-{activation}'
            df.to_csv(f'features_permutation_scores/{model}_{hyper_setting}_scores.csv')

    
        
        

        
        
        
        
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            