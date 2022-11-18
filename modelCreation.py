#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:44:12 2022

@author: francesco
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
tf.random.set_seed(2)

datas = [
 ['SCA_Z500-1_tp-2_dataset'],
 ['NAO_Z500-1_tp-3_dataset'],
 ['EA_MSLP-1_tp-4_dataset'],
 ['EA_MSLP-3_tp-6_dataset'],
 ['ENSO-mei_MSLP-1_tp-7_dataset'],
 ['ENSO-mei_Z500-2_tp-8_dataset'],
 ['SCA_MSLP-3_tp-9_dataset'],
 ['ENSO-mei_Z500-3_tp-10_dataset'],
 ['NAO_MSLP-2_tp-11_dataset'],
 ['NAO_MSLP-2_tp-12_dataset']]

for item in datas:
    tf.keras.backend.clear_session()
    var = item[0]
    temp = var.split('-')[-1]
    temp = '-' + temp
    folder = var.replace(temp,'')


    # dataset load and display
    dataset = pd.read_csv(f'/Users/francesco/Desktop/newNIPA/output/{folder}/{var}.csv')
    
    ### BATCH normalization (pc1_pos and pca_neg normaized together)
    #dataset.loc[:,'pc1']=(dataset.loc[:,'pc1']-dataset.loc[:,'pc1'].mean())/dataset.loc[:,'pc1'].std()
    
    ###SEPARATED normalization (pc1_pos and pca_neg normaized separately)
    dataset.loc[:20,'pc1']=(dataset.loc[:20,'pc1']-dataset.loc[:20,'pc1'].mean())/dataset.loc[:20,'pc1'].std()
    dataset.loc[21:,'pc1']=(dataset.loc[21:,'pc1']-dataset.loc[21:,'pc1'].mean())/dataset.loc[21:,'pc1'].std()
    
    # shuffle the dataset
    dataset = dataset.sample(frac=1)
    
    # division of input variables and target variable
    inp = dataset.loc[:,['pc1','phase_label']]
    out = dataset.loc[:,'target']
    
    split = 35
    
    # split in train and test set
    x_train = inp.to_numpy()[:split]
    y_train = out.to_numpy()[:split]
    
    x_test = inp.to_numpy()[split:]
    y_test = out.to_numpy()[split:]
    
    
    # reshaping data for NN
    x_train = tf.reshape(x_train, (len(x_train),2,1))
    y_train = tf.reshape(y_train, (len(y_train),1))
    
    x_test = tf.reshape(x_test, (len(x_test),2,1))
    y_test = tf.reshape(y_test, (len(y_test),1))
    
    
    # creating NN with functional API
    inputs = tf.keras.layers.Input(shape=(2,1))
    x = tf.keras.layers.Flatten(input_shape=(2,1))(inputs)
    x = tf.keras.layers.Dense(5, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f'{var}')
    model.summary()
    
    # definition of loss and optimizer + compile + fitting
    loss = tf.keras.losses.MSE
    optimizer = tf.keras.optimizers.Adam(0.01)
    model.compile(loss=loss, optimizer=optimizer, metrics=['mean_absolute_error'])
    tf.random.set_seed(2)
    report = model.fit(x=x_train, y=y_train, batch_size=2, epochs=45, validation_data=(x_test,y_test))
    
    MSE_train = report.history['loss'][-1]
    
    
    #model evaluation
    #MSE_val = model.evaluate(x_test,y_test)[0]
    x = tf.reshape(dataset.loc[:,['pc1','phase_label']].to_numpy(), (len(dataset),2,1,1))
    
    result = model.predict(x)
    x_plot = result.squeeze()
      
    x_plot = np.array(x_plot)
    y_plot = dataset['target'].to_numpy()
    
    # computation of pearson coeff btw predicted and observed
    pearson = np.corrcoef(x_plot,y_plot)[0][1]
    
    # computation of r2_score between predicted and observed
    from sklearn.metrics import r2_score
    r2_tot = r2_score(y_plot,x_plot)
    #r2_val = r2_score(y_plot[split:],x_plot[split:])
    
    plt.scatter(x_plot,y_plot)
    plt.title(var)
    plt.xlim(0,130)
    plt.ylim(0,130)
    plt.show()
    
    a = pd.DataFrame(report.history)
    a.plot()
    plt.title(var)
    plt.show()
    
    model.save(f'/Users/francesco/Desktop/NeuralNetworks/models/{var}')
    tf.keras.backend.clear_session()
    del model
    
    
    
    
    
    
    
    
    
    
    
    
