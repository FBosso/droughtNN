#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:15:31 2023

@author: francesco
"""

#import section
import tensorflow as tf
import pandas as pd
import numpy as np
from function_full import params_based_normalization
import matplotlib.pyplot as plt

#needed paths
models_path = '../tuner_trials/2_layer/best'
dataset_name = 'MSSHF-SH-t2m-TCC-tp-UW-VW-MSLP-Z500'
testing_data_path = '../done'

#load model
model = tf.keras.models.load_model(f'{models_path}/{dataset_name}/model')

#load testing data
x_test = pd.read_csv(f'{testing_data_path}/{dataset_name}/x_test_{dataset_name}', index_col = 0)
y_test = pd.read_csv(f'{testing_data_path}/{dataset_name}/y_test_{dataset_name}', index_col = 0)

#load normalization parameters
means = np.load(f'{testing_data_path}/{dataset_name}/means_{dataset_name}.npy')
stds = np.load(f'{testing_data_path}/{dataset_name}/stds_{dataset_name}.npy')


########## prepare TESTING DATA FOR THE ENTIRE PERIOD ##########

#remove unwanted labels form x
entire_x_test = x_test.drop(['beginning_day','beginning_month'],axis=1)
entire_y_test = y_test

#convert testing data into arrays
entire_x_test_array = entire_x_test.to_numpy()
entire_y_test_array = entire_y_test.to_numpy()

#normalize testing data (no target normalization)
entire_x_test_norm = params_based_normalization(entire_x_test_array, means, stds)


########## prepare TESTING DATA MONTH BY MONTH ##########
x_y_test_concat = pd.concat([x_test,y_test], axis=1)
monthly_test = []
for i in range(1,13):
    #select only the training data based on the "pure" month data (average compued only from data of that month)
    test_month_i = x_y_test_concat.loc[(x_test['beginning_day'] == 1) & (x_test['beginning_month'] == i)]
    
    #remove unwanted labels
    test_month_i = test_month_i.drop(['beginning_day','beginning_month'], axis = 1)
    
    #split training and testing
    month_i_x_test = test_month_i.drop(['target'], axis = 1)
    month_i_y_test = test_month_i['target']
    
    #transofrm data into arrays
    month_i_x_test_array = month_i_x_test.to_numpy()
    month_i_y_test_array = month_i_y_test.to_numpy()
    
    #normalize training data
    month_i_x_test_norm = params_based_normalization(month_i_x_test_array, means, stds)
    
    
    #append the tuple of the month i to the list
    monthly_test.append([month_i_x_test_norm,month_i_y_test_array])
    
    
########## TESTING ##########

# ENTIRE
entire = model.evaluate(entire_x_test_norm, entire_y_test_array)
global_tup = (0, entire, 'red')

# MONTHLY
monthly_tests = []
for i,couple in enumerate(monthly_test):
    #extract x test from couple list
    x_test_month_i = couple[0]
    #extract y test from couple list
    y_test_monht_i = couple[1]
    #perform test
    test_value_month_i = model.evaluate(x_test_month_i, y_test_monht_i)
    #store the result in the list
    monthly_tests.append((i+1,test_value_month_i, 'blue'))
    
#crate the summary
summary = monthly_tests
#append global test result at the beginning of the summary
summary.insert(0,global_tup)
#create a dataframe
final_summary = pd.DataFrame(summary, columns=['temp_range','test_MSE','colors'])

#transform test column into array
test_MSE = final_summary['test_MSE'].to_numpy()
x_labels = final_summary['temp_range'].to_numpy()
colors = list(final_summary['colors'])
#create a plot
plt.bar(x_labels,test_MSE, color=colors)
#show all x labels
plt.xticks(x_labels)

plt.ylim(0,1000)


    
    
    





