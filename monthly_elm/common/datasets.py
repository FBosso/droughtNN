#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset generation for the monthly ELM pipeline: builds a per-month feature
matrix from local (.npy) and global (PCA CSV) variable combinations.
"""
import os
import numpy as np
import pandas as pd

from . import config


def timeseries_from_folder(path, month, startyr, endyr):

    files = os.listdir(path)
    files.sort()

    data = []
    for file in files:
        year = int(file.split('-')[0])
        if len(month) == 1:
            month = '0' + str(month)
        if ('-' + month in file) and (year >= startyr) and (year <= endyr):
            data.append(float(np.load(os.path.join(path, file))))
            if len(data) == (42):
                break
    return data


def generate_dataset(month, key):

    datasets = key.split('%')

    loc = []
    glob = []
    for item in datasets:
        if 'csv' in item:
            glob.append(item)
        else:
            loc.append(item)

    if month == '1':
        month = '12'
    else:
        month = str(int(month) - 1)

    loc_data = []
    for item in loc:
        serie = timeseries_from_folder(item, month, 1979, 2021)  # serie dovrà essere una lista
        loc_data.append(serie)  # loc_data dovrà essere una lista di liste

    glob_data = []
    for item in glob:
        glob_data.append(pd.read_csv(item, index_col=0))
    if len(glob_data) > 0:
        if len(glob_data) > 1:
            ordered = []
            for i, d in enumerate(glob_data):
                if i != 0:
                    # maitein target only for one of the datasets
                    d = d.drop('target', axis='columns')
                # sort the values based on the years
                d = d.sort_values('year_glvar')
                # drop the column of the years (needed only for orderinf)
                if i != 0:
                    d = d.drop('year_glvar', axis='columns')
                # save old columns name to modify pc1 (common name between multiple dfs)
                old_columns = list(d.columns)
                # ientify the index of 'pc1'
                index_pc1 = old_columns.index('pc1')
                index_phase = old_columns.index('phase_label')
                # reference 'pc1' nd change its name
                old_columns[index_pc1] = f'pc1_{i}'
                old_columns[index_phase] = f'phase_label_{i}'
                new_columns = old_columns
                # assign renamed columns to df
                d.columns = new_columns
                # concatenate datasets
                d = d.reset_index(drop=True)
                ordered.append(d)
            dataset = pd.concat(ordered, axis=1)

            # only two signal !!!############### no more than two ########
            conditions = [
                (dataset['phase_label_0'] == 1) & (dataset['phase_label_1'] == 1),
                (dataset['phase_label_0'] == 1) & (dataset['phase_label_1'] == 2),
                (dataset['phase_label_0'] == 2) & (dataset['phase_label_1'] == 1),
                (dataset['phase_label_0'] == 2) & (dataset['phase_label_1'] == 2)
            ]
            choices = [1, 2, 3, 4]

            dataset['climate_state'] = np.select(conditions, choices)

            dataset = dataset.drop('phase_label_0', axis='columns')
            dataset = dataset.drop('phase_label_1', axis='columns')

        else:
            dataset = pd.concat(glob_data, axis=1)
            dataset.sort_values('year_glvar')
    else:
        dataset = pd.DataFrame()
        target_path = str(config.LOCAL_DATA_DIR / 'tp')
        if month == '12':
            month_target = '1'
        else:
            month_target = str(int(month) + 1)
        dataset['target'] = timeseries_from_folder(target_path, month_target, 1980, 2021)

    for i, item in enumerate(loc_data):
        variable = loc[i].split('/')[-1]
        dataset[variable] = item

    return dataset
