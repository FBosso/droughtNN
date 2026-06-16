#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model definitions and the Leave-One-Out cross-validation evaluation used to
score every (variable combo, hyperparameter) pair.
"""
import pickle

import numpy as np
import tensorflow as tf
from sklearn import linear_model
from skelm import ELMRegressor

from . import config
from .combo_utils import pretty_combo


@tf.function
def tensorize(x_train, y_train, x_val, y_val):
    # put data into tensors
    x_train = tf.reshape(x_train, (len(x_train), x_train.shape[1], 1))
    y_train = tf.reshape(y_train, (len(y_train), 1))
    x_val = tf.reshape(x_val, (len(x_val), x_train.shape[1], 1))
    y_val = tf.reshape(y_val, (len(y_val), 1))

    return (x_train, y_train, x_val, y_val)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def input_to_hidden(x, Win):
    a = np.dot(x, Win)
    a = np.maximum(a, 0, a)  # ReLU
    return a


def predict(x, Win, Wout):
    x = input_to_hidden(x, Win)
    y = np.dot(x, Wout)
    return y


def LOO_from_dataset(key, data, model_type, month=None, hyperparams=None, save_points=False, already_normalized=False, yr=False, save_LOO_models=False):
    from sklearn.model_selection import LeaveOneOut
    tf.random.set_seed(3)
    '''
    Parameters
    ----------
    key : TYPE string
        DESCRIPTION. '%'-joined paths identifying the variable combination
        used to build `data`.

    data : TYPE pandas.DataFrame
        DESCRIPTION. Dataset produced by generate_dataset().

    model_type : TYPE string
        DESCRIPTION. One of 'NN', 'ELM', 'skELM', 'linear', 'torchNN'.

    Returns: the mean LOO MSE (plus optional points/years depending on flags)
    -------
    '''

    cols = list(data.columns)

    # remove the year of the sample
    if 'year_glvar' in cols:
        data = data.sort_values('year_glvar')
        years = list(data['year_glvar'])
        data = data.drop('year_glvar', axis='columns')

    # remove the 'target' string from the column list to avoid normalization of the target
    cols.remove('target')
    # remove 'year_glvar' string from the column list because related data has already been removed

    try:
        cols.remove('year_glvar')
    except ValueError:
        years = [i for i in range(1980, 2022)]

    try:
        cols.remove('phase_label')
    except ValueError:
        pass

    try:
        cols.remove('climate_state')
    except ValueError:
        pass

    if already_normalized == False:
        # normalize alla the cols named in the list
        for col in cols:
            if not (data[col].max() == data[col].min() == 0):
                data.loc[:, col] = (data.loc[:, col] - data.loc[:, col].mean()) / data.loc[:, col].std()

    # division of input variables and target variable
    inp = data.drop('target', axis='columns').to_numpy()
    out = data.loc[:, 'target'].to_numpy()
    # create a LOO instance
    loo = LeaveOneOut()
    MSEs = []

    val_true = []
    val_hat = []

    for train_index, val_index in loo.split(inp):
        # split the dataset
        x_train = inp[train_index]
        y_train = out[train_index]
        x_val = inp[val_index]
        y_val = out[val_index]

        if model_type == 'NN':

            x_train, y_train, x_val, y_val = tensorize(x_train, y_train, x_val, y_val)

            ############ Change this portion to change the model #############
            # functional API NN creation
            inputs = tf.keras.layers.Input(shape=(inp.shape[1], 1))
            x = tf.keras.layers.Flatten(input_shape=(inp.shape[1], 1))(inputs)
            x = tf.keras.layers.Dense(hyperparams['neuron'], activation=hyperparams['activation'])(x)
            outputs = tf.keras.layers.Dense(1, activation='linear')(x)

            model = tf.keras.Model(inputs=inputs,
                                    outputs=outputs,
                                    name='nn_SCA_MSLP-1_JAN')

            # definition of loss and optimizer
            loss = tf.keras.losses.MSE
            optimizer = tf.keras.optimizers.Adam(hyperparams['learning_rate'])
            # comple the model

            model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=['mean_squared_error'])

            model.fit(x=x_train, y=y_train, batch_size=hyperparams['batch_size'], epochs=hyperparams['epoch'], verbose=0)
            ##################################################################
            MSE = (model(x_val).numpy() - y_val) ** 2
            MSEs.append(MSE)

        elif model_type == 'ELM':

            INPUT_LENGHT = x_train.shape[1]
            HIDDEN_UNITS = hyperparams['neuron']
            valid = False
            while not valid:
                try:
                    # random initialization
                    Win = np.random.normal(size=[INPUT_LENGHT, HIDDEN_UNITS])

                    X = input_to_hidden(x_train, Win)
                    Xt = np.transpose(X)
                    Wout = np.dot(np.linalg.inv(np.dot(Xt, X)), np.dot(Xt, y_train))
                    valid = True
                except np.linalg.LinAlgError:
                    valid = False

            y = predict(x_val, Win, Wout)

            total = y.shape[0]
            for i in range(total):
                MSE = (y_val[i] - y[i]) ** 2
            MSEs.append(MSE)

        elif model_type == 'skELM':

            estimator = ELMRegressor(n_neurons=(hyperparams['neuron']), ufunc=(hyperparams['activation']))
            estimator.fit(x_train, y_train)
            y_hat = estimator.predict(x_val)
            MSEs.append((y_val - y_hat) ** 2)

            combo = pretty_combo(key)

            if save_LOO_models == True:
                hyper_setting = f'neu-{hyperparams["neuron"]}_act-{hyperparams["activation"]}'
                # create combo/month folder if not existing
                combo_dir = config.FEATURES_MODELS_DIR / combo / str(month)
                combo_dir.mkdir(parents=True, exist_ok=True)

                # save model for the current LOO instance
                model_path = combo_dir / f'{model_type}_{hyper_setting}_model_LOO-{val_index[0]}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(estimator, f)

            if save_points == True:
                val_true.append(float(y_val.flat[0]))
                val_hat.append(float(y_hat.flat[0]))

        elif model_type == 'linear':
            regr = linear_model.LinearRegression()
            regr.fit(x_train, y_train)
            y_hat = regr.predict(x_val)
            MSEs.append((y_val - y_hat) ** 2)

            if save_points == True:
                val_true.append(float(y_val.flat[0]))
                val_hat.append(float(y_hat.flat[0]))

        elif model_type == 'torchNN':

            import torch
            import torch.nn as nn

            device = torch.device("cpu")

            class RegressorNN(nn.Module):
                def __init__(self, input_dim, output_dim, hyperparams):
                    super(RegressorNN, self).__init__()
                    self.model = nn.Sequential(
                        nn.Linear(input_dim, hyperparams['neuron']),
                        nn.ReLU(),
                        nn.Linear(hyperparams['neuron'], output_dim)
                    )

                def forward(self, x):
                    return self.model(x)

            input_dim = x_train.shape[1]
            output_dim = 1

            x_train_adapted = x_train.astype(np.float32)
            y_train_adapted = y_train.astype(np.float32).reshape(-1, 1)
            x_val_adapted = x_val.astype(np.float32)
            y_val_adapted = y_val.astype(np.float32).reshape(-1, 1)

            model = RegressorNN(input_dim, output_dim, hyperparams).to(device)
            loss_f = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
            epochs = hyperparams['epoch']

            for epoch in range(epochs):

                inputs = torch.from_numpy(x_train_adapted).requires_grad_().to(device)
                labels = torch.from_numpy(y_train_adapted).to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_f(outputs, labels)
                loss.backward()
                optimizer.step()

                print(epoch, loss.item())

            prediction = model(torch.from_numpy(x_val_adapted)).detach().numpy()[0][0]
            MSE = (prediction - y_val_adapted[0][0]) ** 2
            MSEs.append(MSE)

    MSEs = np.array(MSEs).mean()

    if save_points == True and yr == False:
        return val_true, val_hat, MSEs

    if save_points == True and yr == True:
        return val_true, val_hat, years, MSEs

    return MSEs
