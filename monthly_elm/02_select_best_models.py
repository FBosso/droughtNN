#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 13:42:39 2022

@author: francesco

Stage 2: select the best (combo, neuron, activation) skELM model per month
from the stage-1 LOO scores, fit a linear-regression baseline on the same
data, and save the comparison summary/plots.
"""

# import section
import pandas as pd

from common import config
from common.combo_utils import pretty_combo, create_generation_strings
from common.datasets import generate_dataset
from common.models import LOO_from_dataset
from common.evaluation import best_model_from_csvs
from common.plotting import save_ELM_LIN_plots, save_ELM_obsVSpred_plots

neurons = list(range(6, 13))
activations = ['relu', 'sigm']


############### BEST MODEL SELECTION ###############
dict_results = best_model_from_csvs(neurons, activations)
# create df and save dict as csv
df = pd.DataFrame(dict_results)
df = df[:-1].T
df.columns = ['dataset', 'mse', 'neuron', 'activation']
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(config.BEST_SUMMARY_CSV)

############### GENERATION-STRING CREATION ##################
# create the string to generate the datasets to create the linear models
# based on dataset founded to be the best
gen_strings = create_generation_strings(dict_results)

################# CREATE LINEAR MODEL AND SAVE RESULTS ##################
dict_results_lin = {}
months = list(dict_results.keys())
for month, combo in zip(months, gen_strings):
    # create dataset
    dataset = generate_dataset(str(month), combo)
    # save the LOO MSE of the linear model in the same dictionary containing the results of the ELM
    y_true, y_hat, MSE_lin = LOO_from_dataset(combo, dataset, 'linear', save_points=True, already_normalized=False)
    # store linear results in a dict
    dict_results_lin[month] = (pretty_combo(combo), MSE_lin, (y_true, y_hat))

############### PLOTS CREATION ##################
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ELM vs Linear scatter comparison plot
save_ELM_LIN_plots(dict_results, dict_results_lin, path=config.PLOTS_DIR)

# matrix plot observed vs predicted
save_ELM_obsVSpred_plots(dict_results, path=config.PLOTS_DIR)
