#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:56:52 2024

@author: francesco
"""

#import section
import tensorflow as tf
import pandas as pd
import numpy as np
from function_full import params_based_normalization
import matplotlib.pyplot as plt

#needed paths
models_path = '../2-hyperparameters_tuning/tuner_trials/best_hyperparams'
dataset_name = 'MSSHF-SH-t2m-TCC-tp-UW-VW-MSLP-Z500'
testing_data_path = 'best_data'

#load model
model = tf.keras.models.load_model(f'{models_path}/{dataset_name}/model')

#load testing data
x_train = pd.read_csv(f'{testing_data_path}/{dataset_name}/x_train_{dataset_name}', index_col = 0)
y_train = pd.read_csv(f'{testing_data_path}/{dataset_name}/y_train_{dataset_name}', index_col = 0)

x_test = pd.read_csv(f'{testing_data_path}/{dataset_name}/x_test_{dataset_name}', index_col = 0)
y_test = pd.read_csv(f'{testing_data_path}/{dataset_name}/y_test_{dataset_name}', index_col = 0)

#load normalization parameters
means = np.load(f'{testing_data_path}/{dataset_name}/means_{dataset_name}.npy')
stds = np.load(f'{testing_data_path}/{dataset_name}/stds_{dataset_name}.npy')


########## prepare TESTING DATA FOR THE ENTIRE PERIOD ##########

#remove unwanted labels form x
entire_x_train = x_train.drop(['beginning_day','beginning_month'],axis=1)
entire_y_train = y_train

entire_x_test = x_test.drop(['beginning_day','beginning_month'],axis=1)
entire_y_test = y_test

#convert testing data into arrays
entire_x_train_array = entire_x_train.to_numpy()
entire_y_train_array = entire_y_train.to_numpy()

entire_x_test_array = entire_x_test.to_numpy()
entire_y_test_array = entire_y_test.to_numpy()

#normalize testing data (no target normalization)
entire_x_train_norm = params_based_normalization(entire_x_train_array, means, stds)
entire_x_test_norm = params_based_normalization(entire_x_test_array, means, stds)


# ENTIRE
train_predictions = model.predict(entire_x_train_norm)
train_true = entire_y_train_array

test_predictions = model.predict(entire_x_test_norm)
test_true = entire_y_test_array

#generate and save dfs
df_train_Ys = pd.DataFrame({'true':train_true.squeeze(),
                            'prediction':train_predictions.squeeze()})

df_test_Ys = pd.DataFrame({'true':test_true.squeeze(),
                           'prediction':test_predictions.squeeze()})

df_train_Ys.to_csv('best_predictions/df_train_Ys.csv')
df_test_Ys.to_csv('best_predictions/df_test_Ys.csv')





