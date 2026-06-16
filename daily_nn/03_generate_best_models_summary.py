#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:36:52 2022

@author: francesco

Stage 3: summarize the tuned models from stage 2 by their final validation
loss and rank them.
"""
import pandas as pd

from common import config

# list all the tuned models
models = [p.name for p in config.TUNER_TRIALS_DIR.iterdir() if p.is_dir()]

# loop all models
data = []
for model in models:
    # read the loss data
    loss = pd.read_csv(config.TUNER_TRIALS_DIR / model / 'loss.csv', index_col=0)
    # select the last value of the val loss (because of early stopping at 10 epochs)
    last_val_loss = loss['val_loss'].to_numpy()[-1]
    # build the couple dataset - val_loss
    couple = (model, last_val_loss)
    # append the couple to the data list
    data.append(couple)

# creation of a summary dataframe from the data list
summary = pd.DataFrame(data, columns=['dataset', 'last_val_loss'])
# sorting the data based on last_val_loss
summary = summary.sort_values('last_val_loss')

# save the summary
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(config.SUMMARY_CSV)
