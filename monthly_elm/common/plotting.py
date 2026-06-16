#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting helpers for comparing ELM, linear-regression, and observed
precipitation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config


def save_ELM_LIN_plots(dict_results, dict_results_lin, path):

    config.PREDICTION_VS_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    tot_hat_elm = []
    tot_true_elm = []
    # loop over each month to assign predicted and true points to data structure
    for i in range(1, 13):
        # save predicted vs tested dataset
        data_ELM = {'prediction': np.round(dict_results[i][4][1], 3), 'ground_truth': np.round(dict_results[i][4][0], 3), 'year': np.array(dict_results[i][-1][2])}
        data_lin = {'prediction': np.round(dict_results_lin[i][2][1], 3), 'ground_truth': np.round(dict_results_lin[i][2][0], 3), 'year': np.array(dict_results[i][-1][2])}
        # create dataset with predicted and true points
        df_ELM = pd.DataFrame(data_ELM)
        df_lin = pd.DataFrame(data_lin)
        # save the dataset
        df_ELM.to_csv(config.PREDICTION_VS_TRUTH_DIR / f'{i}_ELM.csv', index=False)
        df_lin.to_csv(config.PREDICTION_VS_TRUTH_DIR / f'{i}_Lin.csv', index=False)
        # double check if saved value of MSE is equal to the result over the saved points (it should)
        # ELM
        mse_points = np.round(np.mean((df_ELM["prediction"].values - df_ELM["ground_truth"].values) ** 2), 3)
        mse_saved = np.round(dict_results[i][1], 3)
        print(f'MSE_saved_ELM {i}: {mse_saved}  |  MSE_recomputed_ELM {i}: {mse_points}')
        # LIN
        mse_points = np.round(np.mean((df_lin["prediction"].values - df_lin["ground_truth"].values) ** 2), 3)
        mse_saved = np.round(dict_results_lin[i][1], 3)
        print(f'MSE_saved_lin {i}: {mse_saved}  |  MSE_recomputed_lin {i}: {mse_points}')

        # create vars for true and predicted ELM points
        true_ELM = df_ELM["ground_truth"].values
        predicted_ELM = df_ELM["prediction"].values
        # create vars for true and predicted Linear points
        true_lin = df_lin["ground_truth"].values
        predicted_lin = df_lin["prediction"].values

        # plot
        plt.scatter(true_ELM, predicted_ELM, label='ELM')
        plt.scatter(true_lin, predicted_lin, label='linear')
        plt.title(f'Month:{i} | PearsonELM:{round(np.corrcoef(true_ELM, predicted_ELM)[0][1], 2)} | PearsonLIN:{round(np.corrcoef(true_lin, predicted_lin)[0][1], 2)}')
        plt.legend()
        plt.xlim(0, 150)
        plt.ylim(0, 150)
        plt.savefig(f'{path}/{i}_ELM-LIN.pdf')
        plt.show()

        tot_true_elm = tot_true_elm + list(true_ELM)
        tot_hat_elm = tot_hat_elm + list(predicted_ELM)

    pearson = np.corrcoef(tot_hat_elm, tot_true_elm)
    print(pearson)


def save_ELM_obsVSpred_plots(dict_results, path):

    months_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }

    tot_hat_elm = []
    tot_true_elm = []

    # prepare the data for the plot
    for i in range(1, 13):
        # create predicted vs tested dataset
        data_ELM = {'prediction': np.round(dict_results[i][4][1], 3), 'ground_truth': np.round(dict_results[i][4][0], 3)}
        # create dataset with predicted and true points
        df_ELM = pd.DataFrame(data_ELM)
        # create vars for true and predicted ELM points
        true_ELM = df_ELM["ground_truth"].values
        predicted_ELM = df_ELM["prediction"].values
        # append values
        tot_true_elm.append(true_ELM)
        tot_hat_elm.append(predicted_ELM)

    # create the matrix plot
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    fig.tight_layout(pad=2.5)
    fig.supxlabel('                    Predicted precipitation [mm]')
    fig.supylabel('Observed precipitation [mm]')

    i = 0
    for ax, hat, true in zip(axs.flat, tot_hat_elm, tot_true_elm):
        i += 1
        ax.set_xlim(-5, 200)
        ax.set_ylim(-5, 200)
        ax.set_title(f'Month: {months_names[i]}')
        ax.scatter(hat, true, alpha=0.5)
        ax.plot([-5, 200], [-5, 200], color='black', linestyle='dashed')

        # save predicted vs tested dataset
        data = {'prediction': np.round(hat, 3), 'ground_truth': np.round(true, 3)}
        df = pd.DataFrame(data)
        mse = np.mean((df["prediction"].values - df["ground_truth"].values) ** 2)
        print(f'MSE {months_names[i]}: {mse}')

    plt.subplots_adjust(left=0.1, bottom=0.1)
    plt.savefig(f'{path}/monthly_test_ELM.pdf')
