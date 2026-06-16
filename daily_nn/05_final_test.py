#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:15:31 2023

@author: francesco

Stage 5: evaluate the best tuned model (results/best_data, populated by
stage 4) globally and month-by-month, saving summary plots and
prediction-vs-truth tables.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from common import config
from common.normalization import params_based_normalization

dataset_name = config.BEST_DATASET_NAME_NN
testing_data_dir = config.BEST_DATA_DIR / dataset_name

# load model
model = tf.keras.models.load_model(config.TUNER_TRIALS_DIR / dataset_name / 'model')

# load testing data
x_test = pd.read_csv(testing_data_dir / f'x_test_{dataset_name}.csv', index_col=0)
y_test = pd.read_csv(testing_data_dir / f'y_test_{dataset_name}.csv', index_col=0)

# load normalization parameters
means = np.load(testing_data_dir / f'means_{dataset_name}.npy')
stds = np.load(testing_data_dir / f'stds_{dataset_name}.npy')


########## prepare TESTING DATA FOR THE ENTIRE PERIOD ##########

# remove unwanted labels form x
entire_x_test = x_test.drop(['beginning_day', 'beginning_month', 'beginning_year'], axis=1)
entire_y_test = y_test

# convert testing data into arrays
entire_x_test_array = entire_x_test.to_numpy()
entire_y_test_array = entire_y_test.to_numpy()

# normalize testing data (no target normalization)
entire_x_test_norm = params_based_normalization(entire_x_test_array, means, stds)


########## prepare TESTING DATA MONTH BY MONTH ##########
x_y_test_concat = pd.concat([x_test, y_test], axis=1)
monthly_test = []
year_lists = []
for i in range(1, 13):
    # select only the training data based on the "pure" month data (average computed only from data of that month)
    test_month_i = x_y_test_concat.loc[(x_test['beginning_day'] == 1) & (x_test['beginning_month'] == i)]

    # save the year column for later
    year_lists.append(test_month_i['beginning_year'].values)

    # remove unwanted labels
    test_month_i = test_month_i.drop(['beginning_day', 'beginning_month', 'beginning_year'], axis=1)

    # split training and testing
    month_i_x_test = test_month_i.drop(['target'], axis=1)
    month_i_y_test = test_month_i['target']

    # transform data into arrays
    month_i_x_test_array = month_i_x_test.to_numpy()
    month_i_y_test_array = month_i_y_test.to_numpy()

    # normalize training data
    month_i_x_test_norm = params_based_normalization(month_i_x_test_array, means, stds)

    # append the tuple of the month i to the list
    monthly_test.append([month_i_x_test_norm, month_i_y_test_array])


########## TESTING ##########

# ENTIRE
entire = model.evaluate(entire_x_test_norm, entire_y_test_array)
global_tup = (0, entire, 'red')

# MONTHLY
monthly_test = [monthly_test[i] for i in range(-1, 11)]
monthly_tests = []
for i, couple in enumerate(monthly_test):
    # extract x test from couple list
    x_test_month_i = couple[0]
    # extract y test from couple list
    y_test_monht_i = couple[1]
    # perform test
    test_value_month_i = model.evaluate(x_test_month_i, y_test_monht_i)
    # store the result in the list
    monthly_tests.append((i + 1, test_value_month_i, 'blue'))

# create the summary
summary = monthly_tests
# append global test result at the beginning of the summary
summary.insert(0, global_tup)
# create a dataframe
final_summary = pd.DataFrame(summary, columns=['temp_range', 'test_MSE', 'colors'])

# transform test column into array
test_MSE = final_summary['test_MSE'].to_numpy()
x_labels = final_summary['temp_range'].to_numpy()
colors = list(final_summary['colors'])


config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
config.PREDICTION_VS_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

# create a plot
plt.bar(x_labels, test_MSE, color=colors)  # to make it work  --> matplotlib 3.2
# show all x labels
plt.xticks(x_labels)
# set label axes
plt.xlabel('month')
plt.ylabel('MSE')
# set title
plt.title('Test MSE: global and by month')
# set y limit for plot
plt.ylim(0, 1100)

# save figure
plt.savefig(config.PLOTS_DIR / 'best_model_test.pdf')


months_names = {
    1: 'Jan',
    2: 'Feb',
    3: 'Mar',
    4: 'Apr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Aug',
    9: 'Sep',
    10: 'Oct',
    11: 'Nov',
    12: 'Dec'
}


fig, axs = plt.subplots(3, 4, figsize=(15, 10))
fig.tight_layout(pad=2.5)
fig.supxlabel('                    Predicted precipitation [mm]')
fig.supylabel('Observed precipitation [mm]')

i = 0
for ax, test, years in zip(axs.flat, monthly_test, year_lists):
    i += 1
    ax.set_xlim(-5, 200)
    ax.set_ylim(-5, 200)
    ax.set_title(f'Month: {months_names[i]}')
    x = model.predict(test[0])
    ax.plot([-5, 200], [-5, 200], color='black', linestyle='dashed')
    ax.scatter(x, test[1], alpha=0.5)

    # save predicted vs tested dataset
    data = {'prediction': np.round(x.reshape(20), 3), 'ground_truth': np.round(test[1], 3), 'years': years}
    df = pd.DataFrame(data)
    df.to_csv(config.PREDICTION_VS_TRUTH_DIR / f'{months_names[i]}.csv', index=False)

plt.subplots_adjust(left=0.1, bottom=0.1)
plt.savefig(config.PLOTS_DIR / 'monthly_test_FFNN.pdf')
