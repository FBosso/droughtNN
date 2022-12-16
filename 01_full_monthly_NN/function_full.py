#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:05:36 2022

@author: francesco
"""
import numpy as np
import pandas as pd
import os


def timeseries_from_folder_full(path,startyr,endyr,target=False):
    '''
    Parameters
    ----------
    path : TYPE str
        DESCRIPTION. path of the folder where all the 
    startyr : TYPE int
        DESCRIPTION. start year to build the timeserie
    endyr : TYPE int
        DESCRIPTION. start year to build the timeserie
    target : TYPE boolean
        DESCRIPTION. to use or not 'target' as name of the column of the df

    Returns a dataframe with a single column representing a timeseries of the specified variable
    -------
    None.

    '''
    files = os.listdir(path)
    files.sort()
    try:
        files.remove('.DS_Store')
    except:
        pass
    data = []
    for file in files:
        if (int(file.split('-')[0]) >= startyr) and (int(file.split('-')[0]) <= endyr):
            data.append(float(np.load(path+'/'+file)))
    var_name = path.split('/')[-1]
    if target == False:
        df = pd.DataFrame(data,columns=[var_name])
    elif target == True:
        df = pd.DataFrame(data,columns=['target'])
    
    return df
    

def generate_full_dataset(startyr,endyr,combo,temp_res='monthly'):
    '''
    Parameters
    ----------
    startyr : TYPE int
        DESCRIPTION. integer indicating the starting year of the target variable of the dataset
    endyr : TYPE int
        DESCRIPTION. integer indicating the ending year of the target variable of the dataset
    combo : TYPE str
        DESCRIPTION. string containing all the input variables path separated by '%'
    temp_res : TYPE string ['monthly' or 'daily']
        DESCRIPTION. string that indicate if the temporal resolution ofthe dataset has to be daily of monthly

    Returns a pandas dataframe representing the dataset (with variables and targhet)
    -------
    None.
    '''
    
    if temp_res == 'monthly':
        inp_variables = combo.split('%')
        timeseries = []
        for item in inp_variables:
            data = timeseries_from_folder_full(item,startyr,endyr)
            timeseries.append(data)
            
        inp = pd.concat(timeseries, axis=1)
        inp = inp.drop(inp.tail(1).index)
        
        target = timeseries_from_folder_full('data/local_data/tp', startyr, endyr, target=True)
        target = target.drop(target.head(1).index)
        target.reset_index(inplace=True, drop=True)
        
        tot = pd.concat([inp,target], axis=1)
        
        return tot
                
        
    elif temp_res == 'daily':
        pass
    
    

def normalize_dataset(dataset):
    '''
    Parameters
    ----------
    dataset : TYPE pandas dataframe
        DESCRIPTION. pandas dataframe with the data to be normalized. It the 
        target is comprised in the dataset passed to this function, its column 
        has to be called 'target'

    Returns the normaized dataset (without normalizing the target)
    -------
    None.

    '''
    columns = list(dataset.columns)
    columns.remove('target')
    
    for col in columns:
        if not(dataset[col].max() == dataset[col].min() == 0):
            dataset.loc[:,col]=(dataset.loc[:,col]-dataset.loc[:,col].mean())/dataset.loc[:,col].std()
            
    
    return dataset


def combo2pretty(combo):
    '''
    Parameters
    ----------
    combo : TYPE str
        DESCRIPTION. combo string composed by the path of all the variables 
        separated by '%' sign

    Returns a string of all the involved variables separated by '-'
    -------
    None.
    '''
    
    paths = combo.split('%')
    variables = ''
    for i,path in enumerate(paths):
        variable = path.split('/')[-1]
        if i != 0:
            variables = variables + '-' + variable
        elif i == 0:
            variables = variables + variable
            
    return variables
        
    
    
    
    
    
    
    
    
    
    
    
    