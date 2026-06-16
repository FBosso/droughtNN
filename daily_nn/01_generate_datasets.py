#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:01:23 2022

@author: francesco

Stage 1: build the local + global (PCA-reduced) daily feature dataset for the
"all variables" combination and split it into train/test sets.
"""
import itertools

import numpy as np
import xarray as xr
from tqdm import tqdm

from common import config
from common.combo_utils import combo2pretty, gen2gens
from common.datasets import (
    generate_full_dataset,
    global_local_corr,
    filtering_conditions,
    reshape_mask2PCA,
    perform_pca,
    random_split,
)

# define starting and ending years of the datasets
startyr = 1979
endyr = 2021
# define training percentage
percentage_train = 0.8
# define minimum correlation threshold for filtering condition on global data
min_corr = 0.0

local_paths = [str(config.LOCAL_DATA_DAILY_DIR / var) for var in config.LOCAL_VARIABLES]
global_paths = [str(config.RAW_GLOBAL_DATA_DIR / var) for var in config.GLOBAL_VARIABLES]

paths = global_paths + local_paths

combos = []
for i in range(9, 10):
    for combination in itertools.combinations(paths, i):
        combos.append(combination)

for combo in tqdm(combos, desc='Datasets creation', leave=True):

    # define the generating string
    gen_str = '%'.join(combo)

    # separate the generating string in 2 generating strings one for local data and one for global data
    local_gen, global_gen = gen2gens(gen_str)
    ###LOCAL###
    if local_gen != '':
        # generate the local dataset (timeseries data)
        dataset = generate_full_dataset(startyr, endyr, local_gen, lead=30, month_label=True, temp_res='moving_monthly_avg')

    # split local data and target
    cols = list(dataset.columns)
    cols.remove('target')
    inp_loc_data = dataset[cols]
    target = dataset['target']
    limit = round(len(inp_loc_data) * percentage_train)

    # randomly divide training and testing data (to avoid to keep sequences)
    x_train_loc, y_train, x_test_loc, y_test, train_boolean_labels = random_split(inp_loc_data, target, limit, even_test=True, temp_res='moving_monthly_avg')

    test_boolean_labels = np.array([not item for item in train_boolean_labels])

    if global_gen != '':
        for item in global_gen.split('%'):
            # detect variable name
            name = item.split('/')[-1]

            # exploit presaved data
            original_dataset = xr.open_dataset(f'{item}.nc', engine='netcdf4')
            adjusted_var = xr.open_dataset(f'{item}_adjusted.nc', engine='netcdf4')
            name = list(adjusted_var.keys())[0]
            adjusted_var = adjusted_var[name]

            ###### GLOBAL DATA ######
            x_train_glob = adjusted_var.data[train_boolean_labels, :, :]
            x_test_glob = adjusted_var.data[test_boolean_labels, :, :]

            # generate correlation map between EACH global variable and the target
            corr_map = global_local_corr(x_train_glob, y_train.to_numpy())
            # apply the filtering conditions to each correlation map (generate the masks)
            mask, area_check_result = filtering_conditions(corr_map, len(y_train), min_corr, original_dataset)
            # reshape maps into matrix (rows --> time, cols --> pixels)
            x_train_glob_reshaped = reshape_mask2PCA(x_train_glob, mask.mask)
            x_test_glob_reshaped = reshape_mask2PCA(x_test_glob, mask.mask)
            # perform the PCA on the training dataset and project the test set in the same space
            train_pc1, test_pc1 = perform_pca(x_train_glob_reshaped, x_test_glob_reshaped)

            # add the global variable to the training set
            x_train_loc = x_train_loc.copy()
            x_train_loc[name] = train_pc1
            # add the global variable to the test set
            x_test_loc = x_test_loc.copy()
            x_test_loc[name] = test_pc1

    # create the dataset ID
    pretty = combo2pretty(gen_str)

dataset_dir = config.GENERATED_DATASETS_DIR / pretty
dataset_dir.mkdir(parents=True, exist_ok=True)
# save training set (both inputs and target)
x_train_loc.to_csv(dataset_dir / f'x_train_{pretty}.csv')
y_train.to_csv(dataset_dir / f'y_train_{pretty}.csv')
# save test set (both inputs and target)
x_test_loc.to_csv(dataset_dir / f'x_test_{pretty}.csv')
y_test.to_csv(dataset_dir / f'y_test_{pretty}.csv')
