#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:54:50 2023

@author: francesco
"""

import matplotlib.pyplot as plt
import numpy as np

#define month
i = 6

#import data
true_years = np.load('years.npy')
true = np.load('true.npy')
ELM = np.load('ELM.npy')
FFNN = np.load('FFNN.npy')
CNN = np.load('CNN.npy')
ECMWF = np.load('ECMWF.npy')


plt.figure(figsize=(12,6))

plt.ylim(0,170)

plt.xlabel(f'Year (month={i})')
plt.ylabel('Cumulative precipitation [mm]')

plt.xticks(true_years)


plt.plot(true_years,true, label='Target (Obserations)', c='black', marker='*')

plt.plot(true_years,ELM, label='ELM Prediction', c='b')
plt.plot(true_years,FFNN, label='FFNN Prediction', c='green')
plt.plot(true_years,CNN, label='CNN Prediction', c='c')
plt.plot(true_years,ECMWF, label='ECMWF Prediction', c='purple')

plt.legend(loc="upper right")
plt.savefig(f'June_comparison.pdf')
plt.show()