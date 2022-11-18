#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:15:09 2022

@author: francesco
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

#############################################


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
        dataset = pd.concat(glob_data, axis=1)
        dataset.sort_values('year_glvar')
    else:
        dataset = pd.DataFrame()
        target_path = '/Users/francesco/Desktop/Università/Magistrale/Matricola/II_anno/Semestre_II/CLINT_project/Tesi/code/local_data/final_data/tp'
        if month == '12':
            month_target = '1'
        else:
            month_target = str(int(month)+1)
        dataset['target'] = timeseries_from_folder(target_path,month_target,1980,2021)
    
    for i,item in enumerate(loc_data):
        variable = loc[i].split('/')[-1]
        dataset[variable] = item
        
    return(dataset)


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


############################################


device = torch.device("cpu")



    



path_index = '/Users/francesco/Desktop/newNIPA/output/NAO_Z500-1_tp/NAO_Z500-1_tp-3_dataset.csv'
path_t2m = '/Users/francesco/Desktop/Università/Magistrale/Matricola/II_anno/Semestre_II/CLINT_project/Tesi/code/local_data/final_data/t2m'
path_tp = '/Users/francesco/Desktop/Università/Magistrale/Matricola/II_anno/Semestre_II/CLINT_project/Tesi/code/local_data/final_data/tp'
key = path_index+'%'+path_t2m+'%'+path_tp
 
data = generate_dataset('1', key)


cols = list(data.columns)
if 'year_glvar' in cols:
    data = data.drop('year_glvar', axis='columns')
    # BATCH normalization (pc1_pos and pca_neg normaized together)
    data.loc[:,'pc1']=(data.loc[:,'pc1']-data.loc[:,'pc1'].mean())/data.loc[:,'pc1'].std()
if 't2m' in cols:
    data.loc[:,'t2m']=(data.loc[:,'t2m']-data.loc[:,'t2m'].mean())/data.loc[:,'t2m'].std()
if 'tp' in cols:
    data.loc[:,'tp']=(data.loc[:,'tp']-data.loc[:,'tp'].mean())/data.loc[:,'tp'].std()



# shuffle the dataset
data = data.sample(frac=1)
# division of input variables and target variable
inp = data.drop('target', axis='columns').to_numpy()
out = data.loc[:,'target'].to_numpy()







from sklearn.model_selection import LeaveOneOut
# create a LOO instance
loo = LeaveOneOut()
MSEs = []
for train_index, val_index in loo.split(inp):
    
    
    class RegressorNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(RegressorNN, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim,7), 
                nn.ReLU(),
                nn.Linear(7,output_dim)
            )
            
        def forward(self, x):
            return self.model(x)
        
    input_dim = inp.shape[1]
    output_dim = 1

    #split the dataset
    x_train = inp[train_index].astype(np.float32)
    y_train = out[train_index].astype(np.float32).reshape(-1,1)
    x_val = inp[val_index].astype(np.float32)
    y_val = out[val_index].astype(np.float32).reshape(-1,1)
    
    model = RegressorNN(input_dim, output_dim).to('cpu')
    loss_f = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    #optimizer = torch.optim.Adam(lr=0.05)
    epochs = 1000
    
    for epoch in tqdm(range(epochs), leave=True):
        
        inputs = torch.from_numpy(x_train).requires_grad_().to('cpu')
        labels = torch.from_numpy(y_train).to('cpu')
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_f(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #print(epoch, loss.item())
    
    import matplotlib.pyplot as plt
    x = model(inputs).detach().numpy()
    y = labels.detach().numpy()
    plt.scatter(x,y)
    plt.scatter(model(torch.from_numpy(x_val)).detach().numpy()[0][0],y_val[0][0])
    plt.xlim(0,130)
    plt.ylim(0,130)
    plt.show()
        
    prediction = model(torch.from_numpy(x_val)).detach().numpy()[0][0]
    MSE = (prediction - y_val[0][0])**2
    MSEs.append(MSE)
print(' ')
print(np.array(MSEs).mean())



from sklearn.metrics import r2_score
print(r2_score(labels.detach().numpy(), model(inputs).detach().numpy()))
from scipy.stats import pearsonr
print(pearsonr(labels.detach().numpy().squeeze(), model(inputs).detach().numpy().squeeze()).statistic)
















