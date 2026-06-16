#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:40:38 2023

@author: francesco

Stage 6: compare the best FFNN's predictions against the ECMWF subseasonal
benchmark for every month, and produce yearly trend plots and cross-pipeline
trend-comparison data.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from common import config
from common.normalization import params_based_normalization
from common.plotting import save_trend_comparison

dataset_name = config.BEST_DATASET_NAME_TREND
testing_data_dir = config.BEST_DATA_DIR / dataset_name

# load model
model = tf.keras.models.load_model(config.TUNER_TRIALS_DIR / dataset_name / 'model')

# load training data
x_train = pd.read_csv(testing_data_dir / f'x_train_{dataset_name}.csv', index_col=0)
y_train = pd.read_csv(testing_data_dir / f'y_train_{dataset_name}.csv', index_col=0)
train = pd.concat([x_train, y_train], axis=1)
train['label'] = 'train'

# load testing data
x_test = pd.read_csv(testing_data_dir / f'x_test_{dataset_name}.csv', index_col=0)
y_test = pd.read_csv(testing_data_dir / f'y_test_{dataset_name}.csv', index_col=0)
test = pd.concat([x_test, y_test], axis=1)
test['label'] = 'test'

# load normalization parameters
means = np.load(testing_data_dir / f'means_{dataset_name}.npy')
stds = np.load(testing_data_dir / f'stds_{dataset_name}.npy')

tot = pd.concat([train, test], axis=0)
tot = tot.sort_index()

base_start_year = 1979
base_end_year = 2021

files = sorted(p.name for p in config.ECMWF_BENCHMARK_DIR.iterdir())

month_day = {1: [12, 3], 2: [12, 31], 3: [1, 31], 4: [3, 3], 5: [3, 31], 6: [5, 2], 7: [6, 1], 8: [7, 2], 9: [8, 3], 10: [8, 31], 11: [10, 1], 12: [11, 2]}

config.TARGETS_DIR.mkdir(parents=True, exist_ok=True)
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

month_dict = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}

ECMWF_MSEs = []
for i, file in enumerate(files):
    # load the file
    ECMWF = np.load(config.ECMWF_BENCHMARK_DIR / file)
    # detect the information in the file's name
    month = int(file.split('_')[0])
    startyr = int(file.split('_')[-1].split('.')[0].split('-')[0])
    endyr = int(file.split('_')[-1].split('.')[0].split('-')[1])
    true_years = np.array([y for y in range(startyr, endyr + 1)])
    # detect the list of true values, predicted values, and years for the specific month

    i = i + 1

    if i == 1 or i == 2 or i == 12:
        base_end_year = 2020
        base_start = startyr - base_start_year
        base_end = base_end_year - endyr
        extremes = tot.loc[(tot['beginning_day'] == month_day[i][1]) & (tot['beginning_month'] == month_day[i][0])]

        start = base_start
        end = len(extremes) - base_end

        extremes = extremes.iloc[start:end]

    else:
        base_start = startyr - base_start_year
        base_end = base_end_year - endyr + 1
        extremes = tot.loc[(tot['beginning_day'] == month_day[i][1]) & (tot['beginning_month'] == month_day[i][0])]

        start = base_start
        end = len(extremes) - base_end

        extremes = extremes.iloc[start:end]

    samples = extremes[['MSSHF', 'SH', 't2m', 'TCC', 'tp', 'UW', 'VW', 'msl', 'z']].to_numpy()
    samples_norm = params_based_normalization(samples, means, stds)

    true = extremes['target'].to_numpy()

    MSE = ((true - ECMWF) ** 2).mean()
    ECMWF_MSEs.append(MSE)

    hat = model.predict(samples_norm)

    # save the data to produce the plot of each month into a prefixed folder (check function to know more)
    save_trend_comparison(i, hat, true, ECMWF, true_years)

    np.save(config.TARGETS_DIR / f'real_ECMWF_target_{i}', true)

    plt.figure(figsize=(12, 6))

    plt.ylim(0, 180)

    plt.xlabel(f'Year (month={month_dict[i]})')
    plt.ylabel('Cumulative precipitation [mm]')

    plt.xticks(true_years)

    X_Y_Spline_true = make_interp_spline(true_years, true)
    X_Y_Spline_elm = make_interp_spline(true_years, hat)
    X_Y_Spline_ECMWF = make_interp_spline(true_years, ECMWF)

    X_true = np.linspace(true_years.min(), true_years.max(), 20)
    Y_true = X_Y_Spline_true(X_true)

    X_elm = np.linspace(true_years.min(), true_years.max(), 20)
    Y_elm = X_Y_Spline_elm(X_elm)

    X_ECMWF = np.linspace(true_years.min(), true_years.max(), 20)
    Y_ECMWF = X_Y_Spline_ECMWF(X_ECMWF)

    plt.plot(X_true, Y_true, label='Obseration', c='black', marker='*')
    plt.plot(X_elm, Y_elm, label='Yearly ML Prediction', c='green')
    plt.plot(X_ECMWF, Y_ECMWF, label='ECMWF Prediction', c='purple')
    plt.legend(loc="upper left")
    plt.ylim(0, 180)
    plt.savefig(config.PLOTS_DIR / f'{i}_trueVSelmVSecmwf.pdf')
