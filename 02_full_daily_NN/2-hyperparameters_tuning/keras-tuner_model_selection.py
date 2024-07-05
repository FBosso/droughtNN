#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 09:37:27 2023

@author: francesco
"""

import tensorflow as tf
import keras_tuner as kt
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from function_full import normalize_dataset, training_based_normalization
import shutil



#define datasets base path
base_path = '../1-generate_datasets/generated_datasets/'
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
    train_x = pd.read_csv(base_path+folder+f'/x_train_{folder}.csv', index_col=0)
    train_y = pd.read_csv(base_path+folder+f'/y_train_{folder}.csv', index_col=0)
    #drop "month" column from x dataset
    train_x = train_x.drop(columns=['beginning_month', 'beginning_day', 'beginning_year'])
    #concatenate input and targets to allow coherent shuffilng
    data = pd.concat([train_x,train_y], axis=1)
    #shuffle training set
    shuffled = data.sample(frac=1)
    #divide data in train_x and train_y
    train_x = shuffled[list(train_x.columns)].to_numpy()
    train_y = shuffled[list(train_y.columns)].to_numpy()
    #define validation percentage
    val_perc = 0.2
    #determine limit training-validation
    limit = round(len(train_x)*(1-val_perc))
    #divide training and validation
    x_train = train_x[:limit,:]
    y_train = train_y[:limit,:]
    x_val = train_x[limit:,:]
    y_val = train_y[limit:,:]
    #normalize training and validation based on mean and std only of the training
    x_train, x_val, means, stds = training_based_normalization(x_train, x_val)
    #save means and stds to reproduce normalization
    np.save(base_path+folder+f'/means_{folder}', means)
    np.save(base_path+folder+f'/stds_{folder}', stds)
    #identify number of features of the current dataset
    cols = train_x.shape[1]
    
    
    def model_builder_2layer(hp):
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(cols,)))
        
        hp_activation = hp.Choice('activation', values=['relu','sigmoid'])
        hp_layer_1 = hp.Int('layer_1', min_value=5, max_value=30, step=1)
        hp_layer_2 = hp.Int('layer_2', min_value=5, max_value=30, step=1)
        hp_learning_rate = hp.Choice('learning_rate', values=[0.001,0.01,0.1])
        
        model.add(tf.keras.layers.Dense(units=hp_layer_1, activation=hp_activation))
        model.add(tf.keras.layers.Dense(units=hp_layer_2, activation=hp_activation))          
        model.add(tf.keras.layers.Dense(1))
        
        optimizer = tf.keras.optimizers.Adam(hp_learning_rate)
        loss = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer = optimizer, loss = loss)
        
        return model
     
    
    tuner2 = kt.Hyperband(model_builder_2layer,
                         objective='val_loss',
                         max_epochs = 200,
                         factor=3,
                         directory='tuner_trials/best_hyperparams',
                         project_name=folder)
    
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    tuner2.search(x_train,y_train, epochs=200, validation_data=(x_val,y_val), callbacks=[stop_early])
    
    best_hps = tuner2.get_best_hyperparameters(1)[0]
    model = tuner2.hypermodel.build(best_hps)
    report = model.fit(x_train,y_train, epochs=200, validation_data=(x_val,y_val), callbacks=[stop_early])
    loss = pd.DataFrame(report.history)
    loss.plot()
    plt.show()
    
    try:
        os.mkdir(f'tuner_trials/best_hyperparams/{folder}')
    except:
        pass
    loss.to_csv(f'tuner_trials/best_hyperparams/{folder}/loss.csv')
    model.save(f'tuner_trials/best_hyperparams/{folder}/model')
    
    try:
        #delete folder in the new location if it already exists
        shutil.rmtree(f'tuned_datasets/{folder}')
    except:
        pass
    #move the folder from generated_datasets to tuned_datasets
    os.rename(base_path+folder, f'tuned_datasets/{folder}')
    
    
    
    