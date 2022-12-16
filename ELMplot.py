#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 08:57:28 2022

@author: francesco
"""
from skelm import ELMRegressor
import numpy as np
from functions_combo import generate_dataset

neurons = 7
month = '3'
combo = '/Users/francesco/Desktop/Università/Magistrale/Matricola/II_anno/Semestre_II/CLINT_project/Tesi/code/local_data/final_data/SD%/Users/francesco/Desktop/Università/Magistrale/Matricola/II_anno/Semestre_II/CLINT_project/Tesi/code/local_data/final_data/UW%/Users/francesco/Desktop/Università/Magistrale/Matricola/II_anno/Semestre_II/CLINT_project/Tesi/code/local_data/final_data/TCC%/Users/francesco/Desktop/newNIPA/output/SCA_MSLP-3_tp/SCA_MSLP-3_tp-4_dataset.csv'
#combo = '/Users/francesco/Desktop/Università/Magistrale/Matricola/II_anno/Semestre_II/CLINT_project/Tesi/code/local_data/final_data/SD%/Users/francesco/Desktop/Università/Magistrale/Matricola/II_anno/Semestre_II/CLINT_project/Tesi/code/local_data/final_data/VW%/Users/francesco/Desktop/Università/Magistrale/Matricola/II_anno/Semestre_II/CLINT_project/Tesi/code/local_data/final_data/TCC%/Users/francesco/Desktop/newNIPA/output/EA_MSLP-1_tp/EA_MSLP-1_tp-4_dataset.csv'

dataset = generate_dataset(month,combo)

estimator = ELMRegressor(n_neurons=(neurons),ufunc=('relu'))
x_train = dataset.drop('target', axis='columns').to_numpy()
y_train = dataset.loc[:,'target'].to_numpy()
estimator.fit(x_train, y_train)
y_hat = estimator.predict(x_train)
import matplotlib.pyplot as plt
plt.scatter(y_hat,y_train)
pearson = np.corrcoef(y_hat,y_train)