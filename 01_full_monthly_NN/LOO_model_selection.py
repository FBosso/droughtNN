#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 14:53:06 2023

@author: francesco
"""

from function_full import normalize_dataset, KFold_normalization
from sklearn.model_selection import KFold
import tensorflow as tf
import pandas as pd
import numpy as np
import os

learning_rates = [0.001,0.01,0.1]
batch_sizes = [1,5,10,15,20]
activation_functions = ['sigmoid','relu']
epochs = [20,25,30,35,40,45,50,55]
neurons = [10,15,20,25,30]
layers = [1,2,3]

#declaration of the dataframe to store the results
results = pd.DataFrame()

#index counter
c = -1

for epoch in epochs:
    for neuron in neurons:
        for batch_size in batch_sizes:
            for layer in layers:
                for learning_rate in learning_rates:
                    for activation_function in activation_functions:
                        #define datasets base path
                        base_path = 'datasets/'
                        #detect available folders
                        folders = os.listdir(base_path)
                        #remove unwanted filenames
                        try:
                            folders.remove('.DS_Store')
                        except:
                            pass
                        
                        #iterate on all the folders containinf the training and testing sets
                        for folder in folders:
                            
                            #load the training set and the target
                            train_x = pd.read_csv(base_path+folder+f'/x_train_{folder}', index_col=0)
                            train_y = pd.read_csv(base_path+folder+f'/y_train_{folder}', index_col=0)
                            #concatenate input and targets to allow coherent shuffilng
                            data = pd.concat([train_x,train_y], axis=1)
                            #shuffle training set
                            shuffled = data.sample(frac=1)
                            #divide data in train_x and train_y
                            train_x = shuffled[list(train_x.columns)].to_numpy()
                            train_y = shuffled[list(train_y.columns)].to_numpy()
                            
                            #create list to store each MSE of each kfold iteration
                            MSEs = []
                            #setup the k-fold procedure
                            kf = KFold(n_splits = 5)
                            for train_index, validation_index in kf.split(train_x):
                                #training data selection
                                training_data_x = train_x[train_index,:]
                                training_data_y = train_y[train_index]
                                #validation data selection
                                validation_data_x = train_x[validation_index,:]
                                validation_data_y = train_y[validation_index]
                                
                                #normalize the dataset (training and validation separately)
                                norm_training_data_x, norm_validation_data_x = KFold_normalization(training_data_x, validation_data_x)
        
                                #create the configuration of the neural network
                                inputs = tf.keras.layers.Input(shape=(train_x.shape[1],))
                                if layer == 1:
                                    x = tf.keras.layers.Dense(neuron,activation=activation_function)(inputs)
                                elif layer == 2:
                                    x = tf.keras.layers.Dense(neuron,activation=activation_function)(inputs)
                                    x = tf.keras.layers.Dense(neuron,activation=activation_function)(x)
                                elif layer == 3:
                                    x = tf.keras.layers.Dense(neuron,activation=activation_function)(inputs)
                                    x = tf.keras.layers.Dense(neuron,activation=activation_function)(x)
                                    x = tf.keras.layers.Dense(neuron,activation=activation_function)(x)
                                
                                outputs = tf.keras.layers.Dense(1)(x)
                                #define the model based on the defined inputs and outpupt
                                model = tf.keras.Model(inputs=inputs, outputs=outputs, name=folder)
                                #define optimizer and loss function
                                optimizer = tf.keras.optimizers.Adam(learning_rate)
                                loss = tf.keras.losses.MSE
                                #compile the model
                                model.compile(optimizer=optimizer, loss=loss)
                                #show the model summary (how much layers, how much neurons per layers,...)
                                model.summary()
                                #train the neural network
                                report = model.fit(norm_training_data_x,training_data_y,batch_size=batch_size,epochs=epoch,validation_data=(validation_data_x,validation_data_y))
                                #compute MSE on validation data
                                MSE = model.evaluate(validation_data_x,validation_data_y)
                                #append mean squared error on validation data to the MSEs list
                                MSEs.append(MSE)
                            
                            #transform list into array
                            MSEs = np.array(MSEs)
                            #compute the mean of all the MSE contained in MSEs
                            final_MSE = MSEs.mean()
                            #increment index
                            c = c + 1
                            #save the setting of the current network
                            configuration = {
                                'index':[c],
                                'variables':[folder],
                                'MSE':[final_MSE], 
                                'epochs':[epoch], 
                                'neurons':[neuron], 
                                'batch_size':[batch_size], 
                                'layers':[layer], 
                                'learning_rate':[learning_rate], 
                                'activation_function':[activation_function]}
                            #creation of the df based on the current setting
                            temp_df = pd.DataFrame(configuration)
                            #index setup
                            temp_df = temp_df.set_index('index')
                            #concatenation of the current df with the global df
                            results = pd.concat([results,temp_df])
                            #save the global df (at each iteration to keep results in case of failure)
                            results.to_csv('model_scores/results.csv')
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
