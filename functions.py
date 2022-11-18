#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:32:52 2022

@author: francesco
"""

import tensorflow as tf
import numpy as np
import itertools

def generate_dataset(combo):
    '''
    

    Parameters
    ----------
    combo : TYPE touple
        DESCRIPTION. tuple containing one of the possible combinations of variable

    Returns dataset with the specified variables combined
    -------
    None.

    '''

@tf.function
def tensorize(x_train,y_train,x_val,y_val):
    #put data into tensors
    x_train = tf.reshape(x_train, (len(x_train),2,1))
    y_train = tf.reshape(y_train, (len(y_train),1))
    x_val = tf.reshape(x_val, (len(x_val),2,1))
    y_val = tf.reshape(y_val, (len(y_val),1))

    return(x_train,y_train,x_val,y_val)


def sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))

def input_to_hidden(x,Win):
    a = np.dot(x, Win)
    #a = np.maximum(a, 0, a) # ReLU
    sigmoid_v = np.vectorize(sigmoid) # sigmoid
    a = sigmoid_v(a)
    return a

def predict(x,Win,Wout):
    x = input_to_hidden(x,Win)
    y = np.dot(x, Wout)
    return y

def loo_from_lists(paths_list,model_type):
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
    LOO_errors = []
    for dataset in paths_list:
        
        # determine all the possible combo of local end global variables
        combo_items = [dataset, 't2m', 'tp']
        combo = []
        for i in range(len(combo_items)):
            for combination in itertools.combinations(combo_items,i):
                combo.append(combination)
                
        #for combination in combo:
        
        # obtain the dataset containing the selected variables
        dataset = generate_dataset(combo)
        
        ##### Funzione che prende il dataset singolo e restituisce quello combinato (se serve)
        
        data = pd.read_csv(dataset)
        # BATCH normalization (pc1_pos and pca_neg normaized together)
        data.loc[:,'pc1']=(data.loc[:,'pc1']-data.loc[:,'pc1'].mean())/data.loc[:,'pc1'].std()
        
        
        
        # shuffle the dataset
        data = data.sample(frac=1)
        # division of input variables and target variable
        inp = data.loc[:,['pc1','phase_label']].to_numpy()
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
                inputs = tf.keras.layers.Input(shape=(2,1))
                x = tf.keras.layers.Flatten(input_shape=(2,1))(inputs)
                x = tf.keras.layers.Dense(5, activation='selu')(x)
                outputs = tf.keras.layers.Dense(1, activation='linear')(x)
                
                model = tf.keras.Model(inputs=inputs, 
                                       outputs=outputs, 
                                       name='nn_SCA_MSLP-1_JAN')
                
                #definition of loss and optimizer
                loss = tf.keras.losses.MSE
                optimizer = tf.keras.optimizers.Adam(0.1)
                #comple the model
    
                model.compile(loss=loss, 
                              optimizer=optimizer, 
                              metrics=['mean_squared_error'])
                
                report = model.fit(x=x_train, y=y_train, batch_size=2, epochs=45, verbose=0)
                ##################################################################
                #the [1] is to select the metric specified in model.compile
                #MSE = model.evaluate(x_val,y_val)[1]
                MSE = (model(x_val).numpy()-y_val)**2
                MSEs.append(MSE)
                
            elif model_type == 'ELM':
                
                INPUT_LENGHT = x_train.shape[1] # 784 
                HIDDEN_UNITS = 10
                valid = False
                while not valid:
                    try:
                        #random initialization
                        #Win = np.random.normal(size=[INPUT_LENGHT, HIDDEN_UNITS])
                        #fixed initialization
                        Win = np.array([[ 1.04528365,  0.32334454,  0.8607634 ,  0.27260735,  0.37658382,
                                -0.36502513, -1.24816169,  0.70330297,  0.10096333, -0.03200629],
                               [ 0.04246666, -1.37914234, -1.30812627,  0.16139536,  0.52091058,
                                 2.07003164, -1.00980336, -0.95902766, -0.19765259,  1.01410628]])
                        
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
            
        print( f' \t Dataset { dataset.split("/")[-1] } \tDONE' )
            
        MSEs = np.array(MSEs).mean()
        LOO_errors.append(MSEs)
    return(LOO_errors)
































