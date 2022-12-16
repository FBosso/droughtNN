#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:28:00 2022

@author: francesco
"""
# import section
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from skelm import ELMRegressor
from sklearn import linear_model

### functions definition ####

def pretty_combo(combo):
    items = combo.split('%')
    key = ''
    for i,item in enumerate(items):
        if i == 0:
            key = key + item.split('/')[-1]
        else:
            key = key + '-' + item.split('/')[-1]
            
    return key

def tuple2key(tup):
    key = ''
    for i,item in enumerate(tup):
        if i > 0:
            key = key + '%' + item
        else:
            key = key + item
    return key


def timeseries_from_folder(path,month,startyr,endyr):
    
    files = os.listdir(path)
    files.sort()
    
    data = []
    for file in files:
        year = int(file.split('-')[0])
        if len(month) == 1:
            month = '0'+str(month)
        if ('-'+month in file) and (year >= startyr) and (year <= endyr):
            data.append(float(np.load(path+'/'+file)))
            if len(data) == (42):
                break
    return data

def generate_dataset(month,key):
    
    datasets = key.split('%')
    
    loc = []
    glob = []
    for item in datasets:
        if 'csv' in item:
            glob.append(item)
        else:
            loc.append(item)
     
    if month == '1':
        month = '12'
    else:
        month = str(int(month)-1)
    
    loc_data = []
    for item in loc:
        serie = timeseries_from_folder(item,month,1979,2021) #serie dovrà essere una lista
        loc_data.append(serie) #loc_data dovrà essere una lista di liste
        
    glob_data = []   
    for item in glob:
        glob_data.append(pd.read_csv(item, index_col=0))
    if len(glob_data) > 0:
        if len(glob_data) > 1:
            ordered = []
            for i,d in enumerate(glob_data):
                if i != 0:
                    # maitein target only for one of the datasets
                    d = d.drop('target', axis='columns')
                # sort the values based on the years
                d = d.sort_values('year_glvar')
                # drop the column of the years (needed only for orderinf)
                if i != 0:
                    d = d.drop('year_glvar', axis='columns')
                # save old columns name to modify pc1 (common name between multiple dfs)
                old_columns = list(d.columns)
                # ientify the index of 'pc1'
                index_pc1 = old_columns.index('pc1')
                index_phase = old_columns.index('phase_label')
                # reference 'pc1' nd change its name
                old_columns[index_pc1] = f'pc1_{i}'
                old_columns[index_phase] = f'phase_label_{i}'
                new_columns = old_columns
                # assign renamed columns to df
                d.columns = new_columns
                # concatenate datasets
                d = d.reset_index(drop=True)
                ordered.append(d)
            dataset = pd.concat(ordered, axis=1)
            
            ## only two signal !!!############### no more than two ########
            conditions = [
                (dataset['phase_label_0'] == 1) & (dataset['phase_label_1'] == 1),
                (dataset['phase_label_0'] == 1) & (dataset['phase_label_1'] == 2),
                (dataset['phase_label_0'] == 2) & (dataset['phase_label_1'] == 1),
                (dataset['phase_label_0'] == 2) & (dataset['phase_label_1'] == 2)
                ]
            choices = [1,2,3,4]
            
            dataset['climate_state'] = np.select(conditions, choices)
            
            dataset = dataset.drop('phase_label_0', axis='columns')
            dataset = dataset.drop('phase_label_1', axis='columns')
            
        else:
            dataset = pd.concat(glob_data, axis=1)
            dataset.sort_values('year_glvar')
    else:
        dataset = pd.DataFrame()
        #target_path = '/Users/francesco/Desktop/Università/Magistrale/Matricola/II_anno/Semestre_II/CLINT_project/Tesi/code/local_data/final_data/tp'
        target_path = 'data/local_data/tp'
        if month == '12':
            month_target = '1'
        else:
            month_target = str(int(month)+1)
        dataset['target'] = timeseries_from_folder(target_path,month_target,1980,2021)
    
    for i,item in enumerate(loc_data):
        variable = loc[i].split('/')[-1]
        dataset[variable] = item
        
    return(dataset)


@tf.function
def tensorize(x_train,y_train,x_val,y_val):
    #put data into tensors
    x_train = tf.reshape(x_train, (len(x_train),x_train.shape[1],1))
    y_train = tf.reshape(y_train, (len(y_train),1))
    x_val = tf.reshape(x_val, (len(x_val),x_train.shape[1],1))
    y_val = tf.reshape(y_val, (len(y_val),1))

    return(x_train,y_train,x_val,y_val)


def sigmoid(x):
    import numpy as np
    return 1 / (1 + np.exp(-x))

def input_to_hidden(x,Win):
    a = np.dot(x, Win)
    a = np.maximum(a, 0, a) # ReLU
    #sigmoid_v = np.vectorize(sigmoid) # sigmoid
    #a = sigmoid_v(a)
    return a

def predict(x,Win,Wout):
    x = input_to_hidden(x,Win)
    y = np.dot(x, Wout)
    return y
    

def LOO_from_dataset(key,data,model_type,hyperparams=None, save_points=False, already_normalized=False, yr=False):
    import pandas as pd
    from sklearn.model_selection import LeaveOneOut
    tf.random.set_seed(3)
    '''
    Parameters
    ----------
    paths_list : TYPE list
        DESCRIPTION. List containing all the paths to all the datasets that we want to compare
        
    model : TYPE string (NN or ELM)
        DESCRIPTION. Describe the tipe of model to build

    Returns: a list containing the LOO validations errors
    -------
    '''

    cols = list(data.columns)
    
    # remove the year of the sample
    if 'year_glvar' in cols:
        data = data.sort_values('year_glvar')
        years = list(data['year_glvar'])
        data = data.drop('year_glvar', axis='columns')
     
    # remove the 'target' string from the column list to avoid normalization of the target
    cols.remove('target')
    # remove 'year_glvar' string from the column list because related data has already been removed
    
    try:
        cols.remove('year_glvar')
    except:
        years = [i for i in range(1980,2022)]
    
    try:
        cols.remove('phase_label')
    except:
        pass
    
    try:
        cols.remove('climate_state')
    except:
        pass
    
    if already_normalized == False:
        # normalize alla the cols named in the list
        for col in cols:
            if not(data[col].max() == data[col].min() == 0):
                data.loc[:,col]=(data.loc[:,col]-data.loc[:,col].mean())/data.loc[:,col].std()
    
    
    # shuffle the dataset
    data = data.sample(frac=1)
    # division of input variables and target variable
    inp = data.drop('target', axis='columns').to_numpy()
    out = data.loc[:,'target'].to_numpy()
    # create a LOO instance
    loo = LeaveOneOut()
    MSEs = []
    
    val_true = []
    val_hat = []
    
    
    for train_index, val_index in loo.split(inp):
        #split the dataset
        x_train = inp[train_index]
        y_train = out[train_index]
        x_val = inp[val_index]
        y_val = out[val_index]
        
        if model_type == 'NN':
        
            x_train,y_train,x_val,y_val = tensorize(x_train,y_train,x_val,y_val)
            
            ############ Change this portion to change the model #############
            #functional API NN creation
            inputs = tf.keras.layers.Input(shape=(inp.shape[1],1))
            x = tf.keras.layers.Flatten(input_shape=(inp.shape[1],1))(inputs)
            x = tf.keras.layers.Dense(hyperparams['neuron'], activation=hyperparams['activation'])(x)
            outputs = tf.keras.layers.Dense(1, activation='linear')(x)
            
            model = tf.keras.Model(inputs=inputs, 
                                   outputs=outputs, 
                                   name='nn_SCA_MSLP-1_JAN')
            
            #definition of loss and optimizer
            loss = tf.keras.losses.MSE
            optimizer = tf.keras.optimizers.Adam(hyperparams['learning_rate'])
            #comple the model

            model.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=['mean_squared_error'])
            
            report = model.fit(x=x_train, y=y_train, batch_size=hyperparams['batch_size'], epochs=hyperparams['epoch'], verbose=0)
            ##################################################################
            #the [1] is to select the metric specified in model.compile
            #MSE = model.evaluate(x_val,y_val)[1]
            MSE = (model(x_val).numpy()-y_val)**2
            MSEs.append(MSE)
            
        elif model_type == 'ELM':
            
            INPUT_LENGHT = x_train.shape[1]
            HIDDEN_UNITS = hyperparams['neuron']
            valid = False
            while not valid:
                try:
                    #random initialization
                    Win = np.random.normal(size=[INPUT_LENGHT, HIDDEN_UNITS])
                    
                    X = input_to_hidden(x_train, Win)
                    Xt = np.transpose(X)
                    Wout = np.dot(np.linalg.inv(np.dot(Xt, X)), np.dot(Xt, y_train))
                    valid = True
                except:
                    valid = False
            
            y = predict(x_val, Win, Wout)
            
            total = y.shape[0]
            for i in range(total):
                MSE = (y_val[i]-y[i])**2
            MSEs.append(MSE)
            
        elif model_type == 'skELM':
            
            estimator = ELMRegressor(n_neurons=(hyperparams['neuron']),ufunc=(hyperparams['activation']))
            estimator.fit(x_train, y_train)
            y_hat = estimator.predict(x_val)
            MSEs.append((y_val-y_hat)**2)
            
            if save_points == True:
                
                val_true.append(float(y_val))
                val_hat.append(float(y_hat))
                
        elif model_type == 'linear':
            regr = linear_model.LinearRegression()
            regr.fit(x_train, y_train)
            y_hat = regr.predict(x_val)
            MSEs.append((y_val-y_hat)**2)
            
            if save_points == True:
                
                val_true.append(float(y_val))
                val_hat.append(float(y_hat))
            
            
        elif model_type == 'torchNN':
            
            import torch
            import torch.nn as nn
            
            device = torch.device("cpu")
            
            class RegressorNN(nn.Module):
                def __init__(self, input_dim, output_dim, hyperparams):
                    super(RegressorNN, self).__init__()
                    self.model = nn.Sequential(
                        nn.Linear(input_dim,hyperparams['neuron']), 
                        nn.ReLU(),
                        nn.Linear(hyperparams['neuron'],output_dim)
                    )
                    
                def forward(self, x):
                    return self.model(x)
                
                
            input_dim = x_train.shape[1]
            output_dim = 1
            
            x_train_adapted = x_train.astype(np.float32)
            y_train_adapted = y_train.astype(np.float32).reshape(-1,1)
            x_val_adapted = x_val.astype(np.float32)
            y_val_adapted = y_val.astype(np.float32).reshape(-1,1)
            
            model = RegressorNN(input_dim, output_dim, hyperparams).to(device)
            loss_f = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
            epochs = hyperparams['epoch']
            
            for epoch in range(epochs):
                
                inputs = torch.from_numpy(x_train_adapted).requires_grad_().to(device)
                labels = torch.from_numpy(y_train_adapted).to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_f(outputs, labels)
                loss.backward()
                optimizer.step()
                
                print(epoch, loss.item())
                #prediction = model(torch.from_numpy(x_val)).to('mps')
                #prediction = prediction.detach().numpy()[0][0]
                #MSE = (prediction - y_val[0][0])**2
                
            prediction = model(torch.from_numpy(x_val_adapted)).detach().numpy()[0][0]
            MSE = (prediction - y_val_adapted[0][0])**2
            MSEs.append(MSE)
            
            
        
    #print( f' \t Dataset { pretty_combo(key) } \tDONE' )
            
    MSEs = np.array(MSEs).mean()
    
    if save_points == True and yr == False:
        return val_true, val_hat
    
    if save_points == True and yr == True:
        return val_true, val_hat, years
        
    return(MSEs)


    
    