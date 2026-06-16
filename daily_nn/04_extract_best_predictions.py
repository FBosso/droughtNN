#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:56:52 2024

@author: francesco

Stage 4: copy the tuned dataset/model selected as best (stage 3 summary) into
results/best_data/, run it through the model, and save train/test
prediction-vs-truth tables.
"""
import shutil

import tensorflow as tf
import pandas as pd
import numpy as np

from common import config
from common.normalization import params_based_normalization

dataset_name = config.BEST_DATASET_NAME_NN

# path to source/destination directories
src_dir = config.TUNED_DATASETS_DIR / dataset_name
dest_dir = config.BEST_DATA_DIR / dataset_name

try:
    shutil.copytree(src_dir, dest_dir)
except FileExistsError:
    pass

# load model
model = tf.keras.models.load_model(config.TUNER_TRIALS_DIR / dataset_name / 'model')

# load testing data
x_train = pd.read_csv(dest_dir / f'x_train_{dataset_name}.csv', index_col=0)
y_train = pd.read_csv(dest_dir / f'y_train_{dataset_name}.csv', index_col=0)

x_test = pd.read_csv(dest_dir / f'x_test_{dataset_name}.csv', index_col=0)
y_test = pd.read_csv(dest_dir / f'y_test_{dataset_name}.csv', index_col=0)

# load normalization parameters
means = np.load(dest_dir / f'means_{dataset_name}.npy')
stds = np.load(dest_dir / f'stds_{dataset_name}.npy')


########## prepare TESTING DATA FOR THE ENTIRE PERIOD ##########

# remove unwanted labels from x
entire_x_train = x_train.drop(['beginning_day', 'beginning_month', 'beginning_year'], axis=1)
entire_y_train = y_train

entire_x_test = x_test.drop(['beginning_day', 'beginning_month', 'beginning_year'], axis=1)
entire_y_test = y_test

# convert testing data into arrays
entire_x_train_array = entire_x_train.to_numpy()
entire_y_train_array = entire_y_train.to_numpy()

entire_x_test_array = entire_x_test.to_numpy()
entire_y_test_array = entire_y_test.to_numpy()

# normalize testing data (no target normalization)
entire_x_train_norm = params_based_normalization(entire_x_train_array, means, stds)
entire_x_test_norm = params_based_normalization(entire_x_test_array, means, stds)


# ENTIRE
train_predictions = model.predict(entire_x_train_norm)
train_true = entire_y_train_array

test_predictions = model.predict(entire_x_test_norm)
test_true = entire_y_test_array

# generate and save dfs
df_train_Ys = pd.DataFrame({'year_input': x_train['beginning_year'].values,
                            'month_input': x_train['beginning_month'].values,
                            'true': train_true.squeeze(),
                            'prediction': train_predictions.squeeze()})

df_test_Ys = pd.DataFrame({'year_input': x_test['beginning_year'].values,
                           'month_input': x_test['beginning_month'].values,
                           'true': test_true.squeeze(),
                           'prediction': test_predictions.squeeze()})

config.BEST_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
df_train_Ys.to_csv(config.BEST_PREDICTIONS_DIR / 'df_train_Ys.csv')
df_test_Ys.to_csv(config.BEST_PREDICTIONS_DIR / 'df_test_Ys.csv')
