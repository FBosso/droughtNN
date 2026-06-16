#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset normalization helpers shared across the daily_nn pipeline.
"""


def normalize_dataset(dataset):
    '''
    Parameters
    ----------
    dataset : TYPE pandas dataframe
        DESCRIPTION. pandas dataframe with the data to be normalized. It the
        target is comprised in the dataset passed to this function, its column
        has to be called 'target'

    Returns the normaized dataset (without normalizing the target)
    -------
    None.

    '''
    columns = list(dataset.columns)
    try:
        columns.remove('target')
    except ValueError:
        pass

    for col in columns:
        if not (dataset[col].max() == dataset[col].min() == 0):
            dataset.loc[:, col] = (dataset.loc[:, col] - dataset.loc[:, col].mean()) / dataset.loc[:, col].std()

    return dataset


def params_based_normalization(data, means, stds):
    '''
    Parameters
    ----------
    data : TYPE numpy array
        DESCRIPTION. data to normalize with respect to mean and st. dev.
    means : TYPE numpy array
        DESCRIPTION. array containing the mean of each feature of the dataset
    stds : TYPE numpy array
        DESCRIPTION. array containing the st. dev. of each feature of the dataset

    Returns the original dataset normalized based on the provided parameters
    -------
    None.

    '''
    n_features = data.shape[1]
    for i in range(n_features):
        # normalize the feature i-esima
        data[:, i] = (data[:, i] - means[i]) / stds[i]

    return data


def training_based_normalization(training, val_or_test):
    '''
    Parameters
    ----------
    training : TYPE numpy array
        DESCRIPTION. training data in form of numpy array
    val_or_test : TYPE numpy array
        DESCRIPTION. validation or test data in form of numpy array

    Returns
    -------
    training: TYPE numpy array
        DESCRIPTION. normalized training set (normalization based on training mean and st_dev)
    val_or_test : TYPE numpy array
        DESCRIPTION. normalized validation set (normalization based on training mean and st_dev)
    '''
    means = []
    stds = []
    n_features = training.shape[1]
    for i in range(n_features):
        # compute mean and st_dev based on training data
        mean = training[:, i].mean()
        st_dev = training[:, i].std()
        # append mean and st_dev into the related lists
        means.append(mean)
        stds.append(st_dev)
        # normalization of training
        training[:, i] = (training[:, i] - mean) / st_dev
        # normalization of validation (based on training mean and st_dev)
        val_or_test[:, i] = (val_or_test[:, i] - mean) / st_dev

    return training, val_or_test, means, stds
