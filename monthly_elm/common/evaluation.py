#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selection of the best-performing (variable combo, hyperparameter) setting per
month from the LOO score/prediction CSVs produced by the dataset-generation
and tuning stage.
"""
import pandas as pd
from tqdm import tqdm

from . import config


def best_model_from_csvs(neu, activations):
    dict_results = {}
    # loop over each month to search the best performing algorithm for each of them
    for i in tqdm(range(1, 13), desc="iterating over months"):
        # initialize list to store all the best values of each file for a specific month
        names = []
        values = []
        neurons = []
        activ = []
        point = []
        # iterate over all the possible number of neurons
        for item in neu:
            # initialize list to store all the best values for each activation functions fo a specific month
            names_temp = []
            values_temp = []
            neurons_temp = []
            activ_temp = []
            points_temp = []
            # iterate over the activation functions
            for activation in activations:
                # define the path of the file based on neurons and activation
                path = config.FEATURES_SCORES_DIR / f'skELM_neu-{item}_act-{activation}_scores.csv'
                path_points = config.FEATURES_PREDICTIONS_DIR / f'skELM_neu-{item}_act-{activation}_predictions.csv'
                # read the file
                a = pd.read_csv(path, index_col=0)
                b = pd.read_csv(path_points, index_col=0, low_memory=False)
                # extract the name of the best performing model in that specific file for the considered month
                name = list(a[str(i)].loc[a[str(i)] == a[str(i)].min()].index)[0]
                # extract the MSE of the best performing model in that specific file for the considered month
                value = a[str(i)].loc[a[str(i)] == a[str(i)].min()].iloc[0]
                # extract the points of the best performing model in that specific file for the considered month
                points = eval(b.loc[name, str(i)])
                # save all the parameters in the temporar list to chose which activation functions overperform the others
                names_temp.append(name)
                values_temp.append(value)
                neurons_temp.append(item)
                activ_temp.append(activation)
                points_temp.append(points)
            # identify the index in the list related to the best performing activation function
            index = values_temp.index(min(values_temp))
            # append all the data of the best performing model to the monthly list
            names.append(names_temp[index])
            values.append(values_temp[index])
            neurons.append(neurons_temp[index])
            activ.append(activ_temp[index])
            point.append(points_temp[index])
        # identify the index of the best performing model in the monthly list
        index = values.index(min(values))
        # store all the info of the model in the dictionary
        dict_results[i] = (names[index], values[index], neurons[index], activ[index], point[index])

    return dict_results
