#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helpers for converting between variable-combination "keys" (paths joined with
'%'), human-readable names, and the dataset-generation strings consumed by
generate_dataset().
"""
from . import config


def pretty_combo(combo):
    items = combo.split('%')
    key = ''
    for i, item in enumerate(items):
        if i == 0:
            key = key + item.split('/')[-1]
        else:
            key = key + '-' + item.split('/')[-1]

    return key


def tuple2key(tup):
    key = ''
    for i, item in enumerate(tup):
        if i > 0:
            key = key + '%' + item
        else:
            key = key + item
    return key


def list2name_NIPA(variables, copy):
    for variable in copy:
        if '.csv' in variable:
            if variables[variables.index(variable) - 3] == 'ENSO':
                ending_index = variables.index(variable)
                variables[ending_index - 3:ending_index + 1] = ['-'.join(variables[ending_index - 3:ending_index + 1])]

            else:
                ending_index = variables.index(variable)
                variables[ending_index - 2:ending_index + 1] = ['-'.join(variables[ending_index - 2:ending_index + 1])]

    return variables


def create_generation_strings(dict_results):

    combos = []
    neurons = []
    activ = []
    for key in dict_results.keys():
        combos.append(dict_results[key][0])
        neurons.append(dict_results[key][2])
        activ.append(dict_results[key][3])

    gen_strings = []
    for combo in combos:
        gen_string = ''
        variables = combo.split('-')
        copy = variables.copy()

        # this function is to re-build the name of the .csv file coming from NIPA from the list "variables"
        variables = list2name_NIPA(variables, copy)

        for i, variable in enumerate(variables):
            if variable.split('.')[-1] == 'csv':
                ending_part = variable.split('-')[-1]
                folder = variable.replace('-' + ending_part, '')
                file = variable
                full_path = str(config.GLOBAL_DATA_DIR / folder / file)

                if i == 0:
                    gen_string = full_path
                elif i != 0:
                    gen_string = gen_string + '%' + full_path

            else:
                full_path = str(config.LOCAL_DATA_DIR / variable)

                if i == 0:
                    gen_string = full_path
                elif i != 0:
                    gen_string = gen_string + '%' + full_path

        gen_strings.append(gen_string)

    return gen_strings
