#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:34:21 2022

@author: francesco
"""

#import section
import itertools
from function_full import tuple2combostring, generate_full_dataset

#definition of the base path of the folders containing the data
local_base_path = '../data/local_data/'
global_base_path = '../data/raw_global_data/'
#definiton of the names of the available variables (folders)
local_variables = ['MER','MSSHF','RH','SD','SH','t2m','TCC','TCWV','tp','UW','VW']
global_variables = ['SST','MSLP','Z500']

#concatenation of base path with folder names
variables = []
for item in local_variables:
    variables.append(local_base_path+item)
for item in global_variables:
    variables.append(global_base_path+item)

#computation of all the possible combinations 
combo =[]
print('\nStart Combo creation ... \n')
#creation of a dictionary with keys reporting all the possibe monthly combo 
#of global and local variables
for j in range(4,10):
    for combination in itertools.combinations(variables,j):
        # take into account only combination with maximum 4 elemens
            combo.append(combination)
print('Combos CREATED')

for item in combo:
    combo_string = tuple2combostring(item)
    dataset = generate_full_dataset(1979, 2021, combo_string)
    