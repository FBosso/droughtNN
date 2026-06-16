#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 09:37:27 2023

@author: francesco

Linear-regression baseline: trained and evaluated on the same dataset used
by the trend-comparison stage (06), for comparison against the FFNN.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

from common import config
from common.normalization import training_based_normalization, params_based_normalization

dataset_name = config.BEST_DATASET_NAME_TREND
data_dir = config.BEST_DATA_DIR / dataset_name

# load the training set and the target
train_x = pd.read_csv(data_dir / f'x_train_{dataset_name}.csv', index_col=0)
train_y = pd.read_csv(data_dir / f'y_train_{dataset_name}.csv', index_col=0)
# drop "month" columns from x dataset
train_x = train_x.drop(columns=['beginning_month', 'beginning_day', 'beginning_year'])
# concatenate input and targets to allow coherent shuffling
data = pd.concat([train_x, train_y], axis=1)
# shuffle training set
shuffled = data.sample(frac=1)
# divide data in train_x and train_y
train_x = shuffled[list(train_x.columns)].to_numpy()
train_y = shuffled[list(train_y.columns)].to_numpy()
# define validation percentage
val_perc = 0.2
# determine limit training-validation
limit = round(len(train_x) * (1 - val_perc))
# divide training and validation
x_train = train_x[:limit, :]
y_train = train_y[:limit, :]
x_val = train_x[limit:, :]
y_val = train_y[limit:, :]
# normalize training and validation based on mean and std only of the training
x_train, x_val, means, stds = training_based_normalization(x_train, x_val)
# save means and stds to reproduce normalization
np.save(data_dir / f'means_{dataset_name}', means)
np.save(data_dir / f'stds_{dataset_name}', stds)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_hat = regr.predict(x_val)
# compute validation MSE
validation_MSE = ((y_val - y_hat) ** 2).mean()


# MONTH BY MONTH TEST
model = regr

# load testing data
x_test = pd.read_csv(data_dir / f'x_test_{dataset_name}.csv', index_col=0)
y_test = pd.read_csv(data_dir / f'y_test_{dataset_name}.csv', index_col=0)

# load normalization parameters
means = np.load(data_dir / f'means_{dataset_name}.npy')
stds = np.load(data_dir / f'stds_{dataset_name}.npy')

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
for i in range(1, 13):
    # select only the training data based on the "pure" month data (average computed only from data of that month)
    test_month_i = x_y_test_concat.loc[(x_test['beginning_day'] == 1) & (x_test['beginning_month'] == i)]

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
entire_y_hat = model.predict(entire_x_test_norm)
entire_test_MSE = ((entire_y_test_array - entire_y_hat) ** 2).mean()

global_tup = (0, entire_test_MSE, 'red')

# MONTHLY
monthly_tests = []
for i, couple in enumerate(monthly_test):
    # extract x test from couple list
    x_test_month_i = couple[0]
    # extract y test from couple list
    y_test_monht_i = couple[1]
    # perform test
    month_i_y_hat = model.predict(x_test_month_i)
    test_value_month_i = ((month_i_y_hat - y_test_monht_i) ** 2).mean()
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

# create a plot
plt.bar(x_labels, test_MSE, color=colors)
# show all x labels
plt.xticks(x_labels)
# set label axes
plt.xlabel('month')
plt.ylabel('MSE')
# set title
plt.title('Test MSE: global and by month - Linear model')
# set y limit for plot
plt.ylim(0, 1800)

# save figure
plt.savefig(config.PLOTS_DIR / 'linear_best_model_test.pdf')
