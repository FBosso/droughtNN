#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:52:44 2022

@author: francesco

Stage 1: enumerate all per-month variable combinations (local + global data),
generate the corresponding datasets, and run Leave-One-Out CV for every
combination across all skELM hyperparameter settings. Writes one LOO-score
CSV per (neuron, activation) setting to results/features_permutation_scores/.
"""

# import section
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import itertools
from concurrent.futures import ProcessPoolExecutor
from common import config
from common.combo_utils import pretty_combo, tuple2key
from common.datasets import generate_dataset
from common.models import LOO_from_dataset
from tqdm import tqdm


def _eval_combo(args):
    """Worker: generate dataset and run LOO for one (month, combo, hyperparams) task."""
    month, combo, neuron, activation, model_type = args
    dataset = generate_dataset(month, combo)
    y_true, y_hat, years, loo = LOO_from_dataset(combo, dataset, model_type,
                                                  hyperparams={'neuron': neuron, 'activation': activation},
                                                  save_points=True, yr=True)
    return loo, y_true, y_hat, years

################ GLOBAL DATA ##################
# identify and load the data of valid combinations per each months
folders_path = config.EXTERNAL_GLOBAL_RAW_DIR
# loading all the dataset paths
files = []
folders = os.listdir(folders_path)
try:
    folders.remove('.DS_Store')
except ValueError:
    pass
# append all the files inside a specific folder to the files list
for folder in folders:
    for file in os.listdir(folders_path / folder):
        if (file.split('_')[-1] == 'dataset.csv') and (file.split('_')[-2].split('-')[0] == 'tp'):
            files.append(str(folders_path / folder / file))
# dividing datasets by moths
months = [i + 1 for i in range(12)]
months_dict = {
    '1': [], '2': [], '3': [], '4': [], '5': [], '6': [],
    '7': [], '8': [], '9': [], '10': [], '11': [], '12': []
}
# store the filename according to the month of the dict
for item in files:
    for month in months:
        if item.split('_')[-2].split('-')[1] == str(month):
            months_dict[str(month)].append(item)


################ LOCAL DATA ##################
# definition of local variables
local_variables = [str(config.LOCAL_DATA_DIR / var) for var in config.LOCAL_VARIABLES]


################ COMBO CREATION ##################
print('Start Combo creation ... \n')
# creation of a dictionary with keys reporting all the possibe monthly combo of global and local variables
for month in months_dict.keys():
    # check if the the considered month has global data coming fron NIPA
    if len(months_dict[month]) > 0:
        # store the local vars in a list
        variables = local_variables.copy()
        # append global variables to the same list of local ones
        for dataset in months_dict[month]:
            variables.append(dataset)
        # initialize list to store combinations
        combo = []
        # generate combinations and store them in the combo list
        for i in range(len(variables)):
            for combination in itertools.combinations(variables, i + 1):
                # take into account only combination with maximum 4 elemens
                if len(combination) <= 4:
                    # consider only the combinations with maximum 1 global signal
                    c = 0
                    for element in combination:
                        if 'csv' in element:
                            c += 1
                    if c <= 2:
                        combo.append(combination)

        keys = []
        for item in combo:

            # transform tuple to concatenation of variable path with separating %
            keys.append(tuple2key(item))
        values = [0 for i in range(len(keys))]
        diz = dict(zip(keys, values))
        months_dict[month] = diz

    else:
        variables = local_variables.copy()
        combo = []
        for i in range(len(variables)):
            for combination in itertools.combinations(variables, i + 1):
                if len(combination) > 0 and len(combination) <= 4:
                    combo.append(combination)
        keys = []
        for item in combo:
            keys.append(tuple2key(item))
        # initialize a zero vector to be replaced with accuracy values later on
        values = [0 for i in range(len(keys))]
        # create a dictionary matching every value to every combination of variable
        diz = dict(zip(keys, values))
        # assign this data structure to the month of the current iteration
        months_dict[month] = diz

    print(f'\tCombo month {month} CREATED')
print('\nCombo Creation DONE\n')


################ MODEL TRAINING ##################
config.FEATURES_SCORES_DIR.mkdir(parents=True, exist_ok=True)
models = ['skELM']
n_workers = os.cpu_count()

for model in models:
    neurons = [6, 7, 8, 9, 10, 11, 12]
    activations = ['relu', 'sigm']
    for neuron in tqdm(neurons, desc='ELM creation', leave=True):
        for activation in activations:
            # flat task list: all (month, combo) pairs for this hyperparameter setting
            tasks = [
                (month, combo, neuron, activation, model)
                for month in months_dict.keys()
                for combo in months_dict[month].keys()
            ]
            # run all tasks in parallel across CPU cores
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(tqdm(
                    executor.map(_eval_combo, tasks, chunksize=20),
                    total=len(tasks),
                    desc=f'  neu={neuron} act={activation}',
                    leave=False,
                ))
            # reassemble results back into months_dict and build predictions_dict
            predictions_dict = {m: {} for m in months_dict.keys()}
            idx = 0
            for month in months_dict.keys():
                for combo in months_dict[month].keys():
                    loo, y_true, y_hat, years = results[idx]
                    months_dict[month][combo] = loo
                    predictions_dict[month][combo] = str((y_true, y_hat, years))
                    idx += 1

            pretty_index = [pretty_combo(item) for item in pd.DataFrame(months_dict).index]
            hyper_setting = f'neu-{neuron}_act-{activation}'

            df_scores = pd.DataFrame(months_dict)
            df_scores = df_scores.set_index(pd.Index(pretty_index))
            df_scores.to_csv(config.FEATURES_SCORES_DIR / f'{model}_{hyper_setting}_scores.csv')

            config.FEATURES_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
            df_pred = pd.DataFrame(predictions_dict)
            df_pred = df_pred.set_index(pd.Index(pretty_index))
            df_pred.to_csv(config.FEATURES_PREDICTIONS_DIR / f'{model}_{hyper_setting}_predictions.csv')
