#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:01:23 2022

@author: francesco
"""

import tensorflow as tf
import numpy as np
from function_full import generate_full_dataset, normalize_dataset, combo2pretty


#create the dataset
test_combo = '/Users/francesco/Desktop/NeuralNetworks/data/local_data/MER%/Users/francesco/Desktop/NeuralNetworks/data/local_data/SH%/Users/francesco/Desktop/NeuralNetworks/data/climate_signals/EA'
dataset = generate_full_dataset(1979, 2021, test_combo)
#normalize the dataset
dataset = normalize_dataset(dataset)

#divide the dataset in train and test and save the test
#shuffle the dataset
dataset = dataset.sample(frac=1,random_state=1)
#convert dataframe into numpy arrays
inp = dataset.drop('target', axis='columns').to_numpy()
out = dataset.loc[:,'target'].to_numpy()
#define the percentage of training and testing
percentage_train = 0.8
limit = round(len(inp)*percentage_train)
#select the training data
x_train = inp[:limit]
y_train = out[:limit]
#select the testing data
x_test = inp[limit:]
y_test = out[limit:]
#save the testing data with the proper name
pretty = combo2pretty(test_combo)
np.save(f'data/test_sets_full/x_test_{pretty}',x_test)
np.save(f'data/test_sets_full/y_test_{pretty}',y_test)

#create the configuration of the neural network
inputs = tf.keras.layers.Input(shape=(3,))
x = tf.keras.layers.Dense(10,activation='relu')(inputs)
x = tf.keras.layers.Dense(10,activation='relu')(x)
x = tf.keras.layers.Dense(10,activation='relu')(x)
outputs = tf.keras.layers.Dense(1)(x)
#define the model based on the defined inputs and outpupt
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Test')
#define optimizer and loss function
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.MSE
#compile the model
model.compile(optimizer=optimizer, loss=loss)
#show the model summary (how much layers, how much neurons per layers,...)
model.summary()
#train the neural network
report = model.fit(x_train,y_train,batch_size=1,epochs=20)

