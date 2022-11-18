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
        dataset = pd.concat(glob_data, axis=1)
        dataset.sort_values('year_glvar')
    else:
        dataset = pd.DataFrame()
        target_path = '/Users/francesco/Desktop/Università/Magistrale/Matricola/II_anno/Semestre_II/CLINT_project/Tesi/code/local_data/final_data/tp'
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
    

def LOO_from_dataset(key,data,model_type,hyperparams):
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
    if 'year_glvar' in cols:
        data = data.drop('year_glvar', axis='columns')
        # BATCH normalization (pc1_pos and pca_neg normaized together)
        data.loc[:,'pc1']=(data.loc[:,'pc1']-data.loc[:,'pc1'].mean())/data.loc[:,'pc1'].std()
    if 't2m' in cols:
        data.loc[:,'t2m']=(data.loc[:,'t2m']-data.loc[:,'t2m'].mean())/data.loc[:,'t2m'].std()
    if 'tp' in cols:
        data.loc[:,'tp']=(data.loc[:,'tp']-data.loc[:,'tp'].mean())/data.loc[:,'tp'].std()
    
    
    
    # shuffle the dataset
    data = data.sample(frac=1)
    # division of input variables and target variable
    inp = data.drop('target', axis='columns').to_numpy()
    out = data.loc[:,'target'].to_numpy()
    # create a LOO instance
    loo = LeaveOneOut()
    MSEs = []
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
            
            estimator = ELMRegressor(n_neurons=(hyperparams['neuron']),ufunc=('relu'))
            estimator.fit(x_train, y_train)
            y_hat = estimator.predict(x_val)
            MSEs.append((y_val-y_hat)**2)
            
            
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
        
    return(MSEs)




    
    