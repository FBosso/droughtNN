#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:01:23 2022

@author: francesco
"""
import os
import tensorflow as tf #
import pandas as pd
import numpy as np
import itertools
import xarray as xr
from tqdm import tqdm
from function_full import generate_full_dataset, normalize_dataset, combo2pretty, gen2gens, global_timeseries_from_folder_full, adjust_global_data, global_local_corr, filtering_conditions, reshape_mask2PCA, perform_pca, gen_signals, random_split

#define starting and ending years of the datasets
startyr = 1979
endyr = 2021
#define training percentage
percentage_train = 0.8
#define minimum correlation threshold for filtering condition on global data
min_corr = 0.1


local_base_path = '../data/local_data/'
global_base_path = '../data/raw_global_data/lead_1_presaved/'
signal_base_path = '../data/climate_signals/'

local_vars = ['MER','MSSHF','RH','SD','SH','t2m','TCC','TCWV','tp','UW','VW']
global_vars = ['MSLP','SST','Z500']
signal_vars = ['EA','SCA','NAO','ENSO']

local_paths = [local_base_path+var for var in local_vars]
global_paths = [global_base_path+var for var in global_vars]
signal_paths = [signal_base_path+var for var in signal_vars]

paths = local_paths + global_paths + signal_paths

combos = []
for i in range(5,11):
    for combination in itertools.combinations(paths,i):
        combos.append(combination)
        
for combo in tqdm(combos, desc='Datasets creation',leave=True):
    #define the generating string
    gen_str = '%'.join(combo)
    
    #separate the generating string in 2 generating strings one for local data and one for global data
    local_gen, global_gen = gen2gens(gen_str)
    sign_gen = gen_signals(gen_str)
    ###LOCAL###
    if local_gen != '':
        #generate the local dataset (timeseries data)
        dataset = generate_full_dataset(startyr, endyr, local_gen, lead=1, month_label = True)
    #load and concatenate the climate signals (if present)
    if sign_gen != '':
        
        if local_gen != '':
            sign_dataset = generate_full_dataset(startyr, endyr, sign_gen, lead=1)
            sign_dataset = sign_dataset.drop(['target'], axis=1)
            dataset = pd.concat([dataset,sign_dataset], axis=1)
            
        elif local_gen == '':
            #no need to remove target because, since there is no local_gen, it is not going to be a duplicate
            dataset = generate_full_dataset(startyr, endyr, sign_gen, lead=1, month_label = True)
            
            
    #split local data and target
    cols = list(dataset.columns)
    cols.remove('target')
    inp_loc_data = dataset[cols]
    target = dataset['target']
    limit = round(len(inp_loc_data)*percentage_train)
    
    #randomly divide training and testing data (to avoid to keep sequences)
    x_train_loc, y_train, x_test_loc, y_test, train_boolean_labels = random_split(inp_loc_data, target, limit, even_test=True)
    test_boolean_labels = np.array([not item for item in train_boolean_labels])
    
    '''
    ###### LOCAL DATA ######
    ##select the training data
    x_train_loc = inp_loc_data.iloc[:limit,:]
    y_train = target.iloc[:limit]
    ##select the testing data
    x_test_loc = inp_loc_data.iloc[limit:,:]
    y_test = target.iloc[limit:]
    '''
    
    if global_gen != '':
        for item in global_gen.split('%'):
            #detect variable name
            name = item.split('/')[-1]
            '''
            #################### 1) online data processing ####################
            #generate the global dataset (gridded data)
            var = global_timeseries_from_folder_full(item,startyr,endyr,lead=1)
            
            #adjust the global data (no normalization because it will be done on the PC1)
            adjusted_var, original_dataset = adjust_global_data(var, subtract_mean=True)
            #split training and testing (both for GLOBAL and LOCAL)
            ###################################################################
            '''
            #################### 2) exploit presaved data #####################
            original_dataset = xr.open_dataset(f'{item}.nc', engine='netcdf4')
            adjusted_var = xr.open_dataset(f'{item}_adjusted.nc', engine='netcdf4')
            name = list(adjusted_var.keys())[0]
            adjusted_var = adjusted_var[name]
            ###################################################################
            
            ###### GLOBA DATA ######
            x_train_glob = adjusted_var.data[train_boolean_labels,:,:]
            x_test_glob = adjusted_var.data[test_boolean_labels,:,:]
        
            #generate correlation map between EACH global variable and the target
            corr_map = global_local_corr(x_train_glob,y_train.to_numpy())
            #apply the filtering conditions to each correlation map (generate the masks)
            mask, area_check_result = filtering_conditions(corr_map, len(y_train), min_corr, original_dataset)
            #reshape maps into matrix (rows --> time, cols --> pixels)
            x_train_glob_reshaped = reshape_mask2PCA(x_train_glob, mask.mask)
            x_test_glob_reshaped = reshape_mask2PCA(x_test_glob, mask.mask)
            #preform the PCA on the training dataset and project the test set in the same space
            train_pc1, test_pc1 = perform_pca(x_train_glob_reshaped, x_test_glob_reshaped)
            
            #add the global variable to the training set
            x_train_loc = x_train_loc.copy()
            x_train_loc[name] = train_pc1
            #add the global variable to the test set
            x_test_loc = x_test_loc.copy()
            x_test_loc[name] = test_pc1
    
    #create the dataset ID
    pretty = combo2pretty(gen_str)
    #make the dataset direcotry
    try:
        os.mkdir(f'datasets/{pretty}')
    except:
        pass
    #save training set (both inputs and target)
    x_train_loc.to_csv(f'datasets/{pretty}/x_train_{pretty}')
    y_train.to_csv(f'datasets/{pretty}/y_train_{pretty}')
    #save test set (both inputs and tagret)
    x_test_loc.to_csv(f'datasets/{pretty}/x_test_{pretty}')
    y_test.to_csv(f'datasets/{pretty}/y_test_{pretty}')
    
    
    '''
    
    #divide the dataset in train and test and save the test
    #shuffle the dataset
    dataset = dataset.sample(frac=1,random_state=1)
    #convert dataframe into numpy arrays
    inp = dataset.drop('target', axis='columns').to_numpy()
    out = dataset.loc[:,'target'].to_numpy()
    #define the percentage of training and testing
    percentage_train = 0.8
    limit = round(len(inp)*percentage_train)
    #select the training data
    x_train = inp[:limit]
    y_train = out[:limit]
    #select the testing data
    x_test = inp[limit:]
    y_test = out[limit:]
    #save the testing data with the proper name
    pretty = combo2pretty(test_combo)
    np.save(f'../data/test_sets_full/x_test_{pretty}',x_test)
    np.save(f'../data/test_sets_full/y_test_{pretty}',y_test)
    
    #separate the training set in two vectros for Local and Global variables
    loc, glob = divide_data(x_train)
    #compute the correlation map between global data and target
    corr_map = global_target_corr(loc, glob)
    #apply the filtering conditions to the just obtained correlation map and obtain the mask
    global_mask = filtering_conditions(corr_map)
    #apply the just obtained mask to the training data
    
    #read the testing data and apply the same mask to the testing data
    
    #re-save the masked testing data
    
    #create the configuration of the neural network
    inputs = tf.keras.layers.Input(shape=(3,))
    x = tf.keras.layers.Dense(10,activation='relu')(inputs)
    x = tf.keras.layers.Dense(10,activation='relu')(x)
    x = tf.keras.layers.Dense(10,activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    #define the model based on the defined inputs and outpupt
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Test')
    #define optimizer and loss function
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MSE
    #compile the model
    model.compile(optimizer=optimizer, loss=loss)
    #show the model summary (how much layers, how much neurons per layers,...)
    model.summary()
    #train the neural network
    report = model.fit(x_train,y_train,batch_size=1,epochs=20)
    
    '''