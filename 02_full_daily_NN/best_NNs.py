#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:36:52 2022

@author: francesco
"""

#import section
import os
import pandas as pd

#define paths
models_path = 'tuner_trials/2_layer/best'
destination_path = 'best_models'

#list all the best models
models = os.listdir(models_path)
#remove unwanted files if present
try:
    models.remove('.DS_Store')
except:
    pass

#loop all models
data = []
for model in models:
    #read the loss data
    loss = pd.read_csv(f'{models_path}/{model}/loss.csv', index_col = 0)
    #select the last value of the val loss (because of early stopping at 10 epochs)
    last_val_loss = loss['val_loss'].to_numpy()[-1]
    #buld the couple dataset - val_loss
    couple = (model,last_val_loss)
    #append the couple to the data list
    data.append(couple)
    
#creation of a summary dataframe from the data list
summary = pd.DataFrame(data,columns=['dataset','last_val_loss'])
#sorting the data based on last_val_loss
summary = summary.sort_values('last_val_loss')

#save the summary
summary.to_csv('best_models/summary.csv')
