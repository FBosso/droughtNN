#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:24:52 2022

@author: francesco
"""

# import section
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# set random seed
#from numpy.random import seed
#seed(1)
#tf.random.set_seed(2)

cols = ['MSE_train','MSE_val','r2_tot','r2_val','pearson']
summary = pd.DataFrame(columns=cols)

learning_rates = [0.01,0.05,0.1]
neurons = [3,4,5,6,7]
batch_sizes = [1,2,3,4,5]
epochs = [8,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
activations = ['relu','selu']

for learning_rate in learning_rates:
    for neuron in neurons:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for activation in activations:
                    
                    
                    # dataset load and display
                    dataset = pd.read_csv('/Users/francesco/Desktop/newNIPA/output/SCA_Z500-1_tp/SCA_Z500-1_tp-2_dataset.csv')

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
                    x = tf.keras.layers.Dense(neuron, activation=activation)(x)
                    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
                    
                    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='nn_SCA_Z500-1_FEB')
                    model.summary()
                    
                    # definition of loss and optimizer + compile + fitting
                    loss = tf.keras.losses.MSE
                    optimizer = tf.keras.optimizers.Adam(learning_rate)
                    model.compile(loss=loss, optimizer=optimizer, metrics=['mean_absolute_error'])
                    tf.random.set_seed(2)
                    report = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epoch)
                    
                    MSE_train = report.history['loss'][-1]
                    
                    
                    #model evaluation
                    MSE_val = model.evaluate(x_test,y_test)[0]
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
                    r2_val = r2_score(y_plot[split:],x_plot[split:])
                    
                    single = pd.DataFrame([[MSE_train,MSE_val,r2_tot,r2_val,pearson]],columns=cols, index=[f'lr-{learning_rate}_neu-{neuron}_bs-{batch_size}_ep-{epoch}_act-{activation}'])
                    dfs = [summary, single]
                    summary = pd.concat(dfs)
                    summary.to_csv('/Users/francesco/Desktop/NeuralNetworks/summary.csv')
                    
summary.to_csv('/Users/francesco/Desktop/NeuralNetworks/summary.csv')
                    


'''
lr-0.05_neu-7_bs-1_ep-20_act-selu

plt.scatter(x_plot,y_plot)
plt.xlim(0,130)
plt.ylim(0,150)
'''







#lr-0.1_neu-6_bs-5_ep-30_act-selu


#lr-0.1_neu-5_bs-2_ep-45_act-selu








