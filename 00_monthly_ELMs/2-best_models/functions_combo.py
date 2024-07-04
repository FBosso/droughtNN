#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:28:00 2022

@author: francesco
"""
# import section
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from skelm import ELMRegressor
from sklearn import linear_model
import os 
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

### functions definition ####

def pretty_combo(combo):
    items = combo.split('%')
    key = ''
    for i,item in enumerate(items):
        if i == 0:
            key = key + item.split('/')[-1]
        else:
            key = key + '-' + item.split('/')[-1]
            
    return key

def tuple2key(tup):
    key = ''
    for i,item in enumerate(tup):
        if i > 0:
            key = key + '%' + item
        else:
            key = key + item
    return key


def timeseries_from_folder(path,month,startyr,endyr):
    
    files = os.listdir(path)
    files.sort()
    
    data = []
    for file in files:
        year = int(file.split('-')[0])
        if len(month) == 1:
            month = '0'+str(month)
        if ('-'+month in file) and (year >= startyr) and (year <= endyr):
            data.append(float(np.load(path+'/'+file)))
            if len(data) == (42):
                break
    return data

def generate_dataset(month,key):
    
    datasets = key.split('%')
    
    loc = []
    glob = []
    for item in datasets:
        if 'csv' in item:
            glob.append(item)
        else:
            loc.append(item)
     
    if month == '1':
        month = '12'
    else:
        month = str(int(month)-1)
    
    loc_data = []
    for item in loc:
        serie = timeseries_from_folder(item,month,1979,2021) #serie dovrà essere una lista
        loc_data.append(serie) #loc_data dovrà essere una lista di liste
        
    glob_data = []   
    for item in glob:
        glob_data.append(pd.read_csv(item, index_col=0))
    if len(glob_data) > 0:
        if len(glob_data) > 1:
            ordered = []
            for i,d in enumerate(glob_data):
                if i != 0:
                    # maitein target only for one of the datasets
                    d = d.drop('target', axis='columns')
                # sort the values based on the years
                d = d.sort_values('year_glvar')
                # drop the column of the years (needed only for orderinf)
                if i != 0:
                    d = d.drop('year_glvar', axis='columns')
                # save old columns name to modify pc1 (common name between multiple dfs)
                old_columns = list(d.columns)
                # ientify the index of 'pc1'
                index_pc1 = old_columns.index('pc1')
                index_phase = old_columns.index('phase_label')
                # reference 'pc1' nd change its name
                old_columns[index_pc1] = f'pc1_{i}'
                old_columns[index_phase] = f'phase_label_{i}'
                new_columns = old_columns
                # assign renamed columns to df
                d.columns = new_columns
                # concatenate datasets
                d = d.reset_index(drop=True)
                ordered.append(d)
            dataset = pd.concat(ordered, axis=1)
            
            ## only two signal !!!############### no more than two ########
            conditions = [
                (dataset['phase_label_0'] == 1) & (dataset['phase_label_1'] == 1),
                (dataset['phase_label_0'] == 1) & (dataset['phase_label_1'] == 2),
                (dataset['phase_label_0'] == 2) & (dataset['phase_label_1'] == 1),
                (dataset['phase_label_0'] == 2) & (dataset['phase_label_1'] == 2)
                ]
            choices = [1,2,3,4]
            
            dataset['climate_state'] = np.select(conditions, choices)
            
            dataset = dataset.drop('phase_label_0', axis='columns')
            dataset = dataset.drop('phase_label_1', axis='columns')
            
        else:
            dataset = pd.concat(glob_data, axis=1)
            dataset.sort_values('year_glvar')
    else:
        dataset = pd.DataFrame()
        #target_path = '/Users/francesco/Desktop/Università/Magistrale/Matricola/II_anno/Semestre_II/CLINT_project/Tesi/code/local_data/final_data/tp'
        target_path = '../../data/local_data/tp'
        if month == '12':
            month_target = '1'
        else:
            month_target = str(int(month)+1)
        dataset['target'] = timeseries_from_folder(target_path,month_target,1980,2021)
    
    for i,item in enumerate(loc_data):
        variable = loc[i].split('/')[-1]
        dataset[variable] = item
        
    return(dataset)


@tf.function
def tensorize(x_train,y_train,x_val,y_val):
    #put data into tensors
    x_train = tf.reshape(x_train, (len(x_train),x_train.shape[1],1))
    y_train = tf.reshape(y_train, (len(y_train),1))
    x_val = tf.reshape(x_val, (len(x_val),x_train.shape[1],1))
    y_val = tf.reshape(y_val, (len(y_val),1))

    return(x_train,y_train,x_val,y_val)


def sigmoid(x):
    import numpy as np
    return 1 / (1 + np.exp(-x))

def input_to_hidden(x,Win):
    a = np.dot(x, Win)
    a = np.maximum(a, 0, a) # ReLU
    #sigmoid_v = np.vectorize(sigmoid) # sigmoid
    #a = sigmoid_v(a)
    return a

def predict(x,Win,Wout):
    x = input_to_hidden(x,Win)
    y = np.dot(x, Wout)
    return y
    

def LOO_from_dataset(key,data,model_type, month=None ,hyperparams=None, save_points=False, already_normalized=False, yr=False, save_LOO_models=False):
    import pandas as pd
    from sklearn.model_selection import LeaveOneOut
    tf.random.set_seed(3)
    '''
    Parameters
    ----------
    paths_list : TYPE list
        DESCRIPTION. List containing all the paths to all the datasets that we want to compare
        
    model : TYPE string (NN or ELM)
        DESCRIPTION. Describe the type of model to build

    Returns: a list containing the LOO validations errors
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
    except:
        years = [i for i in range(1980,2022)]
    
    try:
        cols.remove('phase_label')
    except:
        pass
    
    try:
        cols.remove('climate_state')
    except:
        pass
    
    if already_normalized == False:
        # normalize alla the cols named in the list
        for col in cols:
            if not(data[col].max() == data[col].min() == 0):
                data.loc[:,col]=(data.loc[:,col]-data.loc[:,col].mean())/data.loc[:,col].std()
    
    
    # shuffle the dataset
    #data = data.sample(frac=1)
    # division of input variables and target variable
    inp = data.drop('target', axis='columns').to_numpy()
    out = data.loc[:,'target'].to_numpy()
    # create a LOO instance
    loo = LeaveOneOut()
    MSEs = []
    
    val_true = []
    val_hat = []
    
    
    for train_index, val_index in loo.split(inp):
        #split the dataset
        x_train = inp[train_index]
        y_train = out[train_index]
        x_val = inp[val_index]
        y_val = out[val_index]
        
        if model_type == 'NN':
        
            x_train,y_train,x_val,y_val = tensorize(x_train,y_train,x_val,y_val)
            
            ############ Change this portion to change the model #############
            #functional API NN creation
            inputs = tf.keras.layers.Input(shape=(inp.shape[1],1))
            x = tf.keras.layers.Flatten(input_shape=(inp.shape[1],1))(inputs)
            x = tf.keras.layers.Dense(hyperparams['neuron'], activation=hyperparams['activation'])(x)
            outputs = tf.keras.layers.Dense(1, activation='linear')(x)
            
            model = tf.keras.Model(inputs=inputs, 
                                   outputs=outputs, 
                                   name='nn_SCA_MSLP-1_JAN')
            
            #definition of loss and optimizer
            loss = tf.keras.losses.MSE
            optimizer = tf.keras.optimizers.Adam(hyperparams['learning_rate'])
            #comple the model

            model.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=['mean_squared_error'])
            
            report = model.fit(x=x_train, y=y_train, batch_size=hyperparams['batch_size'], epochs=hyperparams['epoch'], verbose=0)
            ##################################################################
            #the [1] is to select the metric specified in model.compile
            #MSE = model.evaluate(x_val,y_val)[1]
            MSE = (model(x_val).numpy()-y_val)**2
            MSEs.append(MSE)
            
        elif model_type == 'ELM':
            
            INPUT_LENGHT = x_train.shape[1]
            HIDDEN_UNITS = hyperparams['neuron']
            valid = False
            while not valid:
                try:
                    #random initialization
                    Win = np.random.normal(size=[INPUT_LENGHT, HIDDEN_UNITS])
                    
                    X = input_to_hidden(x_train, Win)
                    Xt = np.transpose(X)
                    Wout = np.dot(np.linalg.inv(np.dot(Xt, X)), np.dot(Xt, y_train))
                    valid = True
                except:
                    valid = False
            
            y = predict(x_val, Win, Wout)
            
            total = y.shape[0]
            for i in range(total):
                MSE = (y_val[i]-y[i])**2
            MSEs.append(MSE)
            
        elif model_type == 'skELM':
            
            estimator = ELMRegressor(n_neurons=(hyperparams['neuron']),ufunc=(hyperparams['activation']))
            estimator.fit(x_train, y_train)
            y_hat = estimator.predict(x_val)
            MSEs.append((y_val-y_hat)**2)
            
            combo = pretty_combo(key)
            
            if save_LOO_models == True:
                hyper_setting = f'neu-{hyperparams["neuron"]}_act-{hyperparams["activation"]}'
                #create combo folder if not extisting
                if not os.path.exists(f"features_permutation_models/{combo}"): 
                    os.makedirs(f"features_permutation_models/{combo}")
                #create month subfolder if not existing
                if not os.path.exists(f"features_permutation_models/{combo}/{month}"): 
                    os.makedirs(f"features_permutation_models/{combo}/{month}") 
                
                #save model for the current LOO instance
                with open(f'features_permutation_models/{combo}/{month}/{model_type}_{hyper_setting}_model_LOO-{val_index[0]}.pkl','wb') as f:
                    pickle.dump(estimator,f)
                
            
            if save_points == True:
                
                val_true.append(float(y_val))
                val_hat.append(float(y_hat))
                
        elif model_type == 'linear':
            regr = linear_model.LinearRegression()
            regr.fit(x_train, y_train)
            y_hat = regr.predict(x_val)
            MSEs.append((y_val-y_hat)**2)
            
            if save_points == True:
                
                val_true.append(float(y_val))
                val_hat.append(float(y_hat))
            
            
        elif model_type == 'torchNN':
            
            import torch
            import torch.nn as nn
            
            device = torch.device("cpu")
            
            class RegressorNN(nn.Module):
                def __init__(self, input_dim, output_dim, hyperparams):
                    super(RegressorNN, self).__init__()
                    self.model = nn.Sequential(
                        nn.Linear(input_dim,hyperparams['neuron']), 
                        nn.ReLU(),
                        nn.Linear(hyperparams['neuron'],output_dim)
                    )
                    
                def forward(self, x):
                    return self.model(x)
                
                
            input_dim = x_train.shape[1]
            output_dim = 1
            
            x_train_adapted = x_train.astype(np.float32)
            y_train_adapted = y_train.astype(np.float32).reshape(-1,1)
            x_val_adapted = x_val.astype(np.float32)
            y_val_adapted = y_val.astype(np.float32).reshape(-1,1)
            
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
                #prediction = model(torch.from_numpy(x_val)).to('mps')
                #prediction = prediction.detach().numpy()[0][0]
                #MSE = (prediction - y_val[0][0])**2
                
            prediction = model(torch.from_numpy(x_val_adapted)).detach().numpy()[0][0]
            MSE = (prediction - y_val_adapted[0][0])**2
            MSEs.append(MSE)
            
            
        
    #print( f' \t Dataset { pretty_combo(key) } \tDONE' )
            
    MSEs = np.array(MSEs).mean()
    
    if save_points == True and yr == False:
        return val_true, val_hat, MSEs
    
    if save_points == True and yr == True:
        return val_true, val_hat, years, MSEs
        
    return(MSEs)




def list2name_NIPA(variables, copy):
    for variable in copy:
        if '.csv' in variable:
            if variables[variables.index(variable)-3] == 'ENSO':
                ending_index = variables.index(variable)
                variables[ending_index-3:ending_index+1] = ['-'.join(variables[ending_index-3:ending_index+1])]
                
            else:
                ending_index = variables.index(variable)
                variables[ending_index-2:ending_index+1] = ['-'.join(variables[ending_index-2:ending_index+1])]
                
    return variables




def best_model_from_csvs(neu,activations):
    dict_results = {}
    # loop over each month to search the best performing algorithm for each of them
    for i in tqdm(range(1,13), desc="iterating over months"):
        # initialize list to store all the best values of each file for a specific month
        names = []
        values = []
        neurons = []
        activ = []
        point = []
        # iterate over all the possible number of neurons
        for item in neu:
            # initialize list to store all the best values for each activation functions fo a specific month
            names_temp = []
            values_temp = []
            neurons_temp = []
            activ_temp = []
            points_temp = []
            # iterate over the activation functions
            for activation in activations:
                # define the path of the file based on neurons and activation
                path = f'../1-generate_datasets_&_hyperparameters_tuning/features_permutation_scores/skELM_neu-{item}_act-{activation}_scores.csv'
                path_points = f'../1-generate_datasets_&_hyperparameters_tuning/features_permutation_predictions/skELM_neu-{item}_act-{activation}_predictions.csv'
                # read the file
                a = pd.read_csv(path, index_col=0)
                b = pd.read_csv(path_points, index_col=0, low_memory=False)
                # extract the name of the best performing model in that specific file for the considered month
                name = list(a[str(i)].loc[a[str(i)]==a[str(i)].min()].index)[0]
                # extract the MSE of the best performing model in that specific file for the considered month
                value = a[str(i)].loc[a[str(i)]==a[str(i)].min()].iloc[0]
                # extract the points of the best performing model in that specific file for the considered month
                points = eval(b.loc[name,str(i)])
                # save all the parameters in the temporar list to chose which activation functions overperform the others
                names_temp.append(name)
                values_temp.append(value)
                neurons_temp.append(item)
                activ_temp.append(activation)
                points_temp.append(points)
            # identify the index in the list related to the best performing activation function
            index = values_temp.index(min(values_temp))
            # append all the data of the best performing model to the monthly list
            names.append(names_temp[index])
            values.append(values_temp[index])
            neurons.append(neurons_temp[index])
            activ.append(activ_temp[index])
            point.append(points_temp[index])
        # identify the index of the best performing model in the monthly list 
        index = values.index(min(values))
        # store all the info of the model in the dictionary
        dict_results[i] = (names[index],values[index], neurons[index], activ[index], point[index])
    
    return dict_results



def create_generation_strings(dict_results):
    
    months = list(dict_results.keys())
    combos = []
    neurons = []
    activ = []    
    for key in dict_results.keys():
        combos.append(dict_results[key][0])
        neurons.append(dict_results[key][2])
        activ.append(dict_results[key][3])
        
    gen_strings = []   
    for combo in combos:
        gen_string = ''
        variables = combo.split('-')
        copy = variables.copy()
        
        #this function is to re-build the name of the .csv file coming from NIPA from the list "variables"
        variables = list2name_NIPA(variables, copy)
                
        for i,variable in enumerate(variables):
            if variable.split('.')[-1] == 'csv':
                base_path = '../../data/global_data'
                
                ending_part = variable.split('-')[-1]
                folder = variable.replace('-'+ending_part,'')
                file = variable
                full_path = base_path+'/'+folder+'/'+file
                
                if i == 0:
                    gen_string = full_path
                elif i != 0:
                    gen_string = gen_string+'%'+full_path
                
            else:
                base_path = '../../data/local_data'
                folder = variable
                
                full_path = base_path+'/'+variable
                
                if i == 0:
                    gen_string = full_path
                elif i != 0:
                    gen_string = gen_string+'%'+full_path
        
        gen_strings.append(gen_string)
        
    return gen_strings


def save_ELM_LIN_plots(dict_results, dict_results_lin, path):
    
    tot_hat_elm = []
    tot_true_elm = []
    #loop over each month to assign predicted and true points to data structure
    for i in range(1,13):
        #save predicted vs tested dataset
        data_ELM = {'prediction':np.round(dict_results[i][4][1],3),'ground_truth':np.round(dict_results[i][4][0],3)}
        data_lin = {'prediction':np.round(dict_results_lin[i][2][1],3),'ground_truth':np.round(dict_results_lin[i][2][0],3)}
        #create dataset with predicted and true points
        df_ELM = pd.DataFrame(data_ELM)
        df_lin = pd.DataFrame(data_lin)
        #save the dataset
        df_ELM.to_csv(f'predictionVStruth_datasets/{i}_ELM.csv', index=False)
        df_lin.to_csv(f'predictionVStruth_datasets/{i}_Lin.csv', index=False)
        #double check if saved value of MSE is equal to the result over the saved points (it should)
        ### ELM
        mse_points = np.round(np.mean((df_ELM["prediction"].values-df_ELM["ground_truth"].values)**2),3)
        mse_saved = np.round(dict_results[i][1],3)
        print(f'MSE_saved_ELM {i}: {mse_saved}  |  MSE_recomputed_ELM {i}: {mse_points}')
        ### LIN
        mse_points = np.round(np.mean((df_lin["prediction"].values-df_lin["ground_truth"].values)**2),3)
        mse_saved = np.round(dict_results_lin[i][1],3)
        print(f'MSE_saved_lin {i}: {mse_saved}  |  MSE_recomputed_lin {i}: {mse_points}')
        
        #create vars for true and predicted ELM points
        true_ELM = df_ELM["ground_truth"].values
        predicted_ELM = df_ELM["prediction"].values
        #create vars for true and predicted Linear points
        true_lin = df_lin["ground_truth"].values
        predicted_lin = df_lin["prediction"].values
        
        #plot
        plt.scatter(true_ELM,predicted_ELM,label='ELM')
        plt.scatter(true_lin,predicted_lin,label='linear')
        plt.title(f'Month:{i} | PearsonELM:{round(np.corrcoef(true_ELM,predicted_ELM)[0][1],2)} | PearsonLIN:{round(np.corrcoef(true_lin,predicted_lin)[0][1],2)}')
        plt.legend()
        plt.xlim(0,150)
        plt.ylim(0,150)
        plt.savefig(f'{path}/{i}_ELM-LIN.pdf')
        plt.show()

        tot_true_elm = tot_true_elm + list(true_ELM)
        tot_hat_elm = tot_hat_elm + list(predicted_ELM)

    pearson = np.corrcoef(tot_hat_elm,tot_true_elm)
    print(pearson)
    
    
def save_ELM_obsVSpred_plots(dict_results, path):
    
    months_names = {
        1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
        7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'
        }
    
    tot_hat_elm = []
    tot_true_elm = []
    
    #prepare the data for the plot
    for i in range(1,13):
        #create predicted vs tested dataset
        data_ELM = {'prediction':np.round(dict_results[i][4][1],3),'ground_truth':np.round(dict_results[i][4][0],3)}
        #create dataset with predicted and true points
        df_ELM = pd.DataFrame(data_ELM)
        #create vars for true and predicted ELM points
        true_ELM = df_ELM["ground_truth"].values
        predicted_ELM = df_ELM["prediction"].values
        #append values
        tot_true_elm.append(true_ELM)
        tot_hat_elm.append(predicted_ELM)
    
    #create the matrix plot
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    fig.tight_layout(pad=2.5)
    fig.supxlabel('                    Predicted precipitation [mm]')
    fig.supylabel('Observed precipitation [mm]')

    i = 0
    for ax, hat, true in zip(axs.flat, tot_hat_elm, tot_true_elm):
        i += 1
        ax.set_xlim(-5,200)
        ax.set_ylim(-5,200)
        ax.set_title(f'Month: {months_names[i]}')
        ax.scatter(hat, true, alpha = 0.5)
        ax.plot([-5,200],[-5,200], color='black', linestyle='dashed')
        
        #save predicted vs tested dataset
        data = {'prediction':np.round(hat,3),'ground_truth':np.round(true,3)}
        df = pd.DataFrame(data)
        #df.to_csv(f'predictionVStruth_datasets/{months_names[i]}.csv', index=False)
        mse = np.mean((df["prediction"].values-df["ground_truth"].values)**2)
        print(f'MSE {months_names[i]}: {mse}')
        
    plt.subplots_adjust(left=0.1, bottom=0.1) 
    plt.savefig(f'{path}/monthly_test_ELM.pdf')

    
    