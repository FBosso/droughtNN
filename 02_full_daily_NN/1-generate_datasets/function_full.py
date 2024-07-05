#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:05:36 2022

@author: francesco
"""
import numpy as np
import pandas as pd
import os
import random
import xarray as xr
import tensorflow as tf
import sklearn.decomposition as skd


def timeseries_from_folder_full(path,startyr,endyr,i=1,target=False,month_label=False, temp_res = 'monthly'):
    '''
    Parameters
    ----------
    path : TYPE str
        DESCRIPTION. path of the folder where all the LOCAL (timeseries) data is stored
    startyr : TYPE int
        DESCRIPTION. start year to build the timeserie
    i : TYPE int
        DESCRIPTION. represents if it is the first variable added to the dataset 
        that is under construction or not. If it is the first (i = 0) and 
        month_label is True, then the column related to the month is added. 
        Otherwise not. This is done in order to avoid to have more than one 
        "month" column in the same dataset
    endyr : TYPE int
        DESCRIPTION. end year to build the timeserie
    target : TYPE boolean
        DESCRIPTION. to use or not 'target' as name of the column of the df

    Returns a pandas dataframe with a single column representing a timeseries of the specified variable
    -------
    None.

    '''
    
    
    
    if temp_res == 'monthly':
    
        files = os.listdir(path)
        files.sort()
        try:
            files.remove('.DS_Store')
        except:
            pass
        data = []
        for file in files:
            if (int(file.split('-')[0]) >= startyr) and (int(file.split('-')[0]) <= endyr):
                
                if month_label == True and i==0:
                    #extraxt the month of the sample from the filename (needed to test month by month)
                    month = file.split('-')[1].split('_')[0]
                    data.append([month,float(np.load(path+'/'+file))])
                else:
                    data.append(float(np.load(path+'/'+file)))
                    
        var_name = path.split('/')[-1]
        if target == False:
            
            if month_label == True and i == 0:
                df = pd.DataFrame(data,columns=['month',var_name])
            else:
                df = pd.DataFrame(data,columns=[var_name])
                
        elif target == True:
            df = pd.DataFrame(data,columns=['target'])
        
        return df
    
    
    
    elif temp_res == 'moving_monthly_avg':
        
        files = os.listdir(path)
        files.sort()
        try:
            files.remove('.DS_Store')
        except:
            pass
        data = []
        for file in files:
            if (int(file.split('-')[0]) >= startyr) and (int(file.split('-')[0]) <= endyr):
                
                if month_label == True and i==0:
                    #extraxt the month of the sample from the filename (needed to test month by month)
                    year = file.split('-')[0]
                    month = file.split('-')[1].split('_')[0]
                    day = file.split('-')[2].split('_')[0]
                    data.append([day,month,year,float(np.load(path+'/'+file))])
                else:
                    data.append(float(np.load(path+'/'+file)))
                    
        var_name = path.split('/')[-1]
        if target == False:
            
            if month_label == True and i == 0:
                df = pd.DataFrame(data,columns=['beginning_day','beginning_month','beginning_year',var_name])
                #compute the average over a 30 days windows
                df[var_name] = df[var_name].rolling(30, min_periods=30).mean()
                #shift day and month columns to match them to the average value computed 
                #taking as first value of the moving window exactly the one represented by that day and that month
                df["beginning_day"] = df["beginning_day"].shift(periods=30)
                df["beginning_month"] = df["beginning_month"].shift(periods=30)
                df["beginning_year"] = df["beginning_year"].shift(periods=30)
                #remove NaN values formed by averages computed on a number of values less than 30
            else:
                df = pd.DataFrame(data,columns=[var_name])
                #compute the average over a 30 days windows
                if var_name != 'tp':
                    df[var_name] = df[var_name].rolling(30, min_periods=30).mean()
                elif var_name == 'tp':
                    df[var_name] = df[var_name].rolling(30, min_periods=30).sum()
            
            #remove NaN values formed by averages computed on a number of values less than 30
            df = df.dropna()
                
        elif target == True:
            df = pd.DataFrame(data,columns=['target'])
            df['target'] = df['target'].rolling(30, min_periods=30).sum()
            df = df.dropna()
        
        return df
    

def global_timeseries_from_folder_full(path,startyr,endyr,lead=1, temp_res='monthly'):
    '''
    Parameters
    ----------
    path : TYPE str
        DESCRIPTION. path of the folder where all the GLOBAL data (gridded) is stored (ONLY FOR ONE VARIABLE)
    startyr : TYPE int
        DESCRIPTION. start year to build the timeserie
    endyr : TYPE int
        DESCRIPTION. end year to build the timeserie
    temp_res : TYPE str
        DESCRIPTION. temporal resulution with which the dataset will be generated
    
    Returns a xarray dataframe representing a timeseries of the specified variable (tensor of gridded data)
    -------
    Note: this function works only for one variable ati time, it has to be looped an called multiple times
        if you want to treat more than one variable
    '''
    if temp_res == 'monthly':
        files = os.listdir(path)
        files.sort()
        del files[-lead]
        try:
            files.remove('.DS_Store')
        except:
            pass
        avg_items = []
        for file in files:
            if (int(file.split('-')[0]) >= startyr) and (int(file.split('-')[0]) <= endyr):
                # take each of the considered files and append the monthly mean to the avg_item list
                avg_items.append(xr.open_dataset(f'{path}/{file}', engine='netcdf4').mean(dim='time'))
        # concatenate the monthly means on the time dimension
        data = xr.concat(avg_items, dim='time')
        
        return data
    
    
    elif temp_res == 'moving_monthly_avg':
        
        files = os.listdir(path)
        files.sort()
        #del files[-lead]
        try:
            files.remove('.DS_Store')
        except:
            pass
        items = []
        for file in files:
            if (int(file.split('-')[0]) >= startyr) and (int(file.split('-')[0]) <= endyr):
                # take each of the considered files and append the monthly mean to the avg_item list
                items.append(xr.open_dataset(f'{path}/{file}', engine='netcdf4'))
        # concatenate the monthly means on the time dimension
        data = xr.concat(items, dim='time')
        new_data = data.where(data.time != data.time[-1], drop=True)
        #Adjust global data (transform and center with respect to the mean)
        new_data, original_dataset = adjust_global_data(new_data, subtract_mean=True)
        #replace NaN values with -99999 (or rolling does not work)
        new_data = new_data.fillna(-99999)
        #apply the moving average 30 days window
        moving_window_avg = new_data.rolling(time=30, min_periods=30).mean(dim='time').dropna("time")
        #replace -99999 with np.nan again (or land masking will not work)
        moving_window_avg = moving_window_avg.where(moving_window_avg != -99999)
        
        #var = list(moving_window_avg.keys())[0]
        true_array = [True for i in range(len(moving_window_avg)-lead)]
        false_array = [False for i in range(lead)]
        total_array = true_array + false_array
        #drop the last 30 samples to build the dataset with the lead time
        data_shifted = moving_window_avg.where(moving_window_avg.time[total_array], drop=True)
        
        return data_shifted, original_dataset



def adjust_global_data(variable, subtract_mean = False):
    '''
    Parameters
    ----------
    variable : TYPE xarray dataset
        DESCRIPTION. xarray dataset containing the data related to a specific variable to be adjusted

    Returns the adjusted dataset in terms of specific variable (e.g. Z500 is dovoded by 9.80665) 
        and in term of mean subtracted (if subtract_mean = True)
    -------
    None.
    '''
    var = list(variable.keys())[0]
    fieldData = variable[var]
    
    
    if var == 'z':
        #finding geopotential height from geopotential by dividing for 
        #gravity acceleration --> (m^2 / s^2) / (m / s^2)
        fieldData = fieldData/9.80665
    elif var == 'msl':
        #converting MSLP from Pa to hPa (because of coherence with 
        #geopotential that is in hPa)
        fieldData = fieldData/100
    if subtract_mean == True:
        #subtract mean pixel by pixel (over the time dimension) to find the anomalies
        for i in range(fieldData.shape[1]):
            for j in range(fieldData.shape[2]):
                fieldData[:,i,j] = fieldData[:,i,j] - fieldData[:,i,j].mean()
                
    return fieldData, variable


    
def significance_level_filtering(r, n, twotailed = True, corrconf = 0.95):
    '''
    Parameters
    ----------
    r : TYPE numpy array
        DESCRIPTION. correlation map
    n : TYPE int
        DESCRIPTION. len of the data on which the correlation map was computed
    twotailed : TYPE, optional
        DESCRIPTION. The default is True.

    Returns a correlation map filtered on 95% significance level (all pixels 
        with a lower significance level will be masked out)
    -------
    None.
    '''
    #### Function copied from NIPA module ###
    import numpy as np
    from scipy.stats import t as tdist
    df = n - 2

	# Create t-statistic
	# Use absolute value to be able to deal with negative scores
    t = np.abs(r * np.sqrt(df/(1-r**2)))
    p = (1 - tdist.cdf(t,df))
    if twotailed:
         p * 2
    #save the p value in a variable
    p_value = p
    #compute the correlation level
    corrlevel = 1 - corrconf
    #Prepare significance level mask
    significance_mask =  (~(p_value < corrlevel))
    #Mask insignificant gridpoints
    corr_grid = np.ma.masked_array(r, significance_mask)
        
    return corr_grid


def land_masking(corr_map):
    '''
    Parameters
    ----------
    corr_map : TYPE numpy array
        DESCRIPTION. correlation map

    Returns a correlation map filtered on NaN values (all NaN values will be masked out)
    -------
    None.
    '''
    
    corr_grid = np.ma.masked_array(corr_map, np.isnan(corr_map))
    
    return corr_grid


def correlation_threshold_filtering(corr_map,min_corr):
    '''
    Parameters
    ----------
    corr_map : TYPE numpy array
        DESCRIPTION. correlation map
    min_corr : TYPE float
        DESCRIPTION. minimum accptable correlation threshold 
        (in both directions + and -)

    Returns a correlation map filtered on minimum correlation values (all values 
        not eough correlated wil be masked out)
    -------
    None.
    '''
    
    #Prepare correlation mask
    corr_mask = (corr_map >-min_corr) & (corr_map < min_corr)
    #Mask not highly correlated gridpoints
    corr_grid = np.ma.masked_array(corr_map, corr_mask)
    
    return corr_grid


def reshape_mask2PCA(full_data, mask):
    '''
    Parameters
    ----------
    full_data : TYPE numpy array
        DESCRIPTION. numpy array containing all the global data (not 
                masked yet, still on global extension)
    mask : TYPE numpy array
        DESCRIPTION. boolean array containing True where the pixels have 
        to be masked and False where the pixels have not to be masked

    Returns the multidimensional data (n_tim, n_lat, n_lon) reshaped in 
    a form which is compatible with PCA (matrix with n_cols = n_pixels 
    and n_rows = n_tim. In other words, each column represents a 
    specific pixel and contains all the value of that pixel along the 
    time dimension)
    -------
    None.
    '''
    #the real mask has to be opposite because the True values are the ones that will be considered (and not masked)
    opposite_mask = mask == False
    data = full_data[:,opposite_mask]
    
    return data



def north_south_masking(corr_map, data_with_LatLon):
    '''
    Parameters
    ----------
    corr_map : TYPE numpy array
        DESCRIPTION. correlation map
    data_with_LatLon : TYPE xarray dataset
        DESCRIPTION. complete dataset with latitude and logitude attached in 
            order to be able to mask the pixels at the extreme latitudes

    Returns a correlation map filtered on latitude (latitudes > 60 and < -30 
        will be masked out)
    -------
    None.
    '''
    
    var = list(data_with_LatLon.keys())[0]
    
    #Mask northern/southern ocean
    corr_map.mask[data_with_LatLon[var].lat > 60] = True
    corr_map.mask[data_with_LatLon[var].lat < -30] = True
    
    return corr_map

def area_checking(corr_grid, original_data):
    '''
    Parameters
    ----------
    corr_grid : TYPE numpy array
        DESCRIPTION. correlation map
    original_data : TYPE xarray dataset
        DESCRIPTION. original dataset (needed to keep the extent of the 
        not filtered map and generate the conovlution operation to 
        perform the area check procedure)

    Returns
    -------
    int
        DESCRIPTION. 1 if area check is passed 0 if not

    '''
    
    var = list(original_data.keys())[0]
    #Check area with convolutions
    # boolean map to be checked
    h = original_data[var].shape[1]
    le = original_data[var].shape[2]
    x_in = (~corr_grid.mask).astype(int).reshape(1,h,le,1)
    x = tf.constant(x_in, dtype=tf.float32)
    # convolution kernel definition
    l = 3
    kernel_in = np.ones((l,l,1,1)) # [filter_height, filter_width, in_channels, out_channels]
    kernel = tf.constant(kernel_in, dtype=tf.float32)
    # execution of the convolutions
    result = tf.nn.conv2d(x, kernel, strides=[1, l, l, 1], padding='VALID').numpy()
    
    # check
    
    if (l**2 not in result):
        return 0
    elif (l**2 in result):
        return 1
    
    

def filtering_conditions(corr_map,timeseries_len,min_corr,original_data):
    
    '''
    Parameters
    ----------
    corr_map : TYPE numpy array (grid)
        DESCRIPTION. a correlation map
    timeseries_len : TYPE int
        DESCRIPTION. the length on which the correlation map was computed

    Returns a masked correlation map based on different conditions
    -------
    None.
    '''
    
    sign_level_filtered = significance_level_filtering(corr_map, timeseries_len)
    #it works only if NaN values are founded, it will not affect variables that are not SST (or similar)
    land_masked = land_masking(sign_level_filtered)
    #corr_theshold_filtered = correlation_threshold_filtering(land_masked, min_corr)
    #reassigning var just to skip correlation rfiltering step
    corr_threshold_filtered = land_masked
    #masking extreme north an extreme south areas
    north_south_masked = north_south_masking(corr_threshold_filtered, original_data)
    area_checked = area_checking(north_south_masked, original_data)    
    
    return north_south_masked, area_checked



def perform_pca(train,test):
    '''
    Parameters
    ----------
    train : TYPE numpy array
        DESCRIPTION. training data on whiche the PCA have to be performed
    test : TYPE numpy array
        DESCRIPTION. testing data to be reprojected in the same PCA space 
        of the training data

    Returns
    -------
    pc1_train : TYPE numpy array
        DESCRIPTION. first principal component for the training set
    pc1_test : TYPE numpy array
        DESCRIPTION. first principal component for the test set
    '''
    
    pca = skd.PCA(n_components = 1)
    pca.fit_transform(train)
    pc1_train = pca.transform(train)
    pc1_test = pca.transform(test)
    
    return pc1_train, pc1_test

def global_local_corr(X,y):
    '''
    Parameters
    ----------
    X : TYPE numpy array (ntim, nlat, nlon)
        DESCRIPTION. Gridded data, 3 dimensions (ntim, nlat, nlon)
    y : TYPE numpy array (ntim)
        DESCRIPTION. Time series, 1 dimension (ntim)

    Returns correlation maps between local precipitation and global variable
    -------
    None.
    '''
    
    #### Function copied from NIPA module ###
    
    # Function to correlate a single time series with a gridded data field
    # X - Gridded data, 3 dimensions (ntim, nlat, nlon)
    # Y - Time series, 1 dimension (ntim)
    ntim, nlat, nlon = X.shape
    ngrid = nlat * nlon
    
    y = y.reshape(1, ntim)
    X = X.reshape(ntim, ngrid).T
    Xm = X.mean(axis = 1).reshape(ngrid,1)
    ym = y.mean()
    r_num = np.sum((X-Xm) * (y-ym), axis = 1)
    r_den = np.sqrt(np.sum((X-Xm)**2, axis = 1) * np.sum((y-ym)**2))
    r = (r_num/r_den).reshape(nlat, nlon)
    
    return r
            
            


def generate_full_dataset(startyr,endyr,combo,temp_res='monthly',lead=1, month_label = False):
    '''
    Parameters
    ----------
    startyr : TYPE int
        DESCRIPTION. integer indicating the starting year of the target variable of the dataset
    endyr : TYPE int
        DESCRIPTION. integer indicating the ending year of the target variable of the dataset
    combo : TYPE str
        DESCRIPTION. string containing all the input variables path separated by '%'
    temp_res : TYPE string ['monthly' or 'daily']
        DESCRIPTION. string that indicate if the temporal resolution ofthe dataset has to be daily of monthly
    month_label : TYPE boolean
        DESCRIPTION. boolean value to indicate if the output dataset must have a column representing the month 
        of the specific sample (to test the different algorithms month by months we need to know which samples 
        belong to which month)

    Returns a pandas dataframe representing the dataset (with variables and target)
    -------
    None.
    '''
    
    #extract global path from combo (global_vars) and separate the generating string of local data (combo)
    global_vars, combo = detect_global(combo)
    
    
    if temp_res == 'monthly':
        inp_variables = combo.split('%')
        timeseries = []
        for i,item in enumerate(inp_variables):
            iteration = i
            data = timeseries_from_folder_full(item,startyr,endyr,iteration, month_label = month_label, temp_res = temp_res)
            timeseries.append(data)
            
        inp = pd.concat(timeseries, axis=1)
        #remove the last sample of the input to create the leadtime of one month with the target
        inp = inp.drop(inp.tail(lead).index)
        
        target = timeseries_from_folder_full('../data/local_data/tp', startyr, endyr, target=True, temp_res = temp_res)
        #remove the first sample of the target to create the leadtime of one month with the input
        target = target.drop(target.head(lead).index)
        target.reset_index(inplace=True, drop=True)
        
        tot = pd.concat([inp,target], axis=1)
        
        return tot
                
        
    elif temp_res == 'moving_monthly_avg':
        inp_variables = combo.split('%')
        timeseries = []
        for i,item in enumerate(inp_variables):
            iteration = i
            #temp_res declares how data should be transformed
            data = timeseries_from_folder_full(item,startyr,endyr,iteration, month_label = month_label, temp_res = temp_res)
            timeseries.append(data)
            
        inp = pd.concat(timeseries, axis=1)
        #remove the last sample of the input to create the leadtime of one month with the target
        inp = inp.drop(inp.tail(lead).index)
        inp.reset_index(inplace=True, drop=True)
        target = timeseries_from_folder_full('../../data/local_data_daily/tp', startyr, endyr, target=True, temp_res = temp_res)
        #remove the first sample of the target to create the leadtime of one month with the input
        target = target.drop(target.head(lead).index)
        target.reset_index(inplace=True, drop=True)
        
        tot = pd.concat([inp,target], axis=1)
        
        return tot
    
    
def random_split(local_input, target, limit, even_test=False, temp_res = 'monthly'):
    '''
    Parameters
    ----------
    local_input : TYPE pandas DataFrame
        DESCRIPTION. dataframe containing the local variables of the considered combo
    target : TYPE pandas DataFrame
        DESCRIPTION. dataframe containing the target of the considered combo
    limit : TYPE int
        DESCRIPTION. integer representing the nimber of sampes to consider 
        for the training set
    even_test : TYPE boolean
        DESCRIPTION. boolean representing if the test set have to be equally distributed 
        over all the 12 months (same number of randomly extracted test samples for each month)

    Returns
    -------
    x_train_loc : TYPE pandas DataFrame
        DESCRIPTION. shuffled coherent extraction of training input data
    y_train : TYPE pandas DataFrame
        DESCRIPTION. shuffled coherent extraction of training target data
    x_test_loc : pandas DataFrame
        DESCRIPTION. shuffled coherent extraction of testing input data
    y_test : TYPE numpy array
        DESCRIPTION. shuffled coherent extraction of testing target data
    tot_idx : TYPE numpy array
        DESCRIPTION. boolean array to select the global data with the 
        same pattern as the local ones
    '''
    
    if temp_res == 'monthly':
    
        if even_test == False:
        
            #list with number of True equal to the number of training samples
            train_keep = [True for i in range(limit)]
            #list with number of False equal to the number of testing samples
            train_not_keep = [False for i in range(len(target)-limit)]
            #concatenate the two lists
            tot_idx = pd.Series(train_keep + train_not_keep)
            #shuffle the list
            random.shuffle(tot_idx)
            #generate opposite list (to select test data)
            opposite_tot_idx = pd.Series([not idx for idx in tot_idx])
            
            #create outputs
            x_train_loc = local_input[tot_idx.values]
            y_train = target[tot_idx.values]
            x_test_loc = local_input[opposite_tot_idx.values]
            y_test = target[opposite_tot_idx.values]
            
            tot_idx = tot_idx.to_numpy()
            
            
            return x_train_loc, y_train, x_test_loc, y_test, tot_idx
        
        elif even_test == True:
            
            #adjust limit value to produce a test set that can be eavenly divided on the 12 months
            #we want to test the models month by month (same number of test samples)
            while (len(target)-limit)%12 != 0:
                #conservative choice, we increment the number of test samples until we reach a number that can be divided by 12
                limit = limit - 1
            #
            test_per_month = (len(target)-limit)/12
            #declare list of months
            months = ['01','02','03','04','05','06','07','08','09','10','11','12']
            idx_test = np.array([])
            for month in months:
                #extract the samples of the specific month
                df_month = local_input.loc[local_input['month'] == month].copy()
                #make the index as a new column
                index = list(df_month.index)
                df_month['index'] = index
                #extract the test samples for the specific month
                samples = df_month.sample(n=int(test_per_month))
                #extract the indexes of the samples
                indexes = samples['index'].to_numpy()
                #concatenate arrays
                idx_test = np.concatenate([idx_test,indexes])
                
            idx_test = idx_test.astype(int)
            boolean_pattern = [True for i in range(len(target))]
            replacements = [False for i in range(len(idx_test))]
            
            for (index, replacement) in zip(idx_test, replacements):
                boolean_pattern[index] = replacement
            
                opposite_boolean_pattern = [not idx for idx in boolean_pattern]
            
            #create outputs    
            x_train_loc = local_input[boolean_pattern]
            y_train = target[boolean_pattern]
            x_test_loc = local_input[opposite_boolean_pattern]
            y_test = target[opposite_boolean_pattern]
            
            return x_train_loc, y_train, x_test_loc, y_test, boolean_pattern
    
    elif temp_res == 'moving_monthly_avg':
        
        #adjust limit value to produce a test set that can be eavenly divided on the 12 months
        #we want to test the models month by month (same number of test samples)
        while (len(target)-limit)%12 != 0:
            #conservative choice, we increment the number of test samples until we reach a number that can be divided by 12
            limit = limit - 1
        test_per_month = (len(target)-limit)/12
        test_pure_month = 20
        #declare list of months
        months = ['01','02','03','04','05','06','07','08','09','10','11','12']
        idx_test = np.array([])
        for month in months:
            #extract the samples of the specific month
            df_month = local_input.loc[(local_input['beginning_month'] == month) & (local_input['beginning_day'] == '01')].copy()
            #
            df_month_not_day = local_input.loc[(local_input['beginning_month'] == month) & (local_input['beginning_day'] != '01')].copy()
            #make the index as a new column
            index = list(df_month.index)
            df_month['index'] = index
            index2 = list(df_month_not_day.index)
            df_month_not_day['index'] = index2
            #extract the test samples for the specific month (PURE)
            samples = df_month.sample(n=int(test_pure_month))
            #extract the test samples for the specific month (NOT PURE)
            samples2 = df_month_not_day.sample(n=int(test_per_month - test_pure_month))
            #extract the indexes of the samples (PURE)
            indexes = samples['index'].to_numpy()
            #extract the indexes of the samples (NOT PURE)
            indexes2 = samples2['index'].to_numpy()
            #concatenate arrays
            idx_test = np.concatenate([idx_test,indexes,indexes2])
            
        idx_test = idx_test.astype(int)
        boolean_pattern = [True for i in range(len(target))]
        replacements = [False for i in range(len(idx_test))]
        
        for (index, replacement) in zip(idx_test, replacements):
            boolean_pattern[index] = replacement
            #reverse the training boolean pattern in order to extract the test data
            opposite_boolean_pattern = [not idx for idx in boolean_pattern]
        
        #create outputs
        x_train_loc = local_input[boolean_pattern]
        y_train = target[boolean_pattern]
        x_test_loc = local_input[opposite_boolean_pattern]
        y_test = target[opposite_boolean_pattern]
        
        return x_train_loc, y_train, x_test_loc, y_test, boolean_pattern
    
  
############################ EXPERIMENTAL ############################


def sequential_split(local_input, target, limit, even_test=False, temp_res = 'moving_monthly_avg'):
    '''
    Parameters
    ----------
    local_input : TYPE pandas DataFrame
        DESCRIPTION. dataframe containing the local variables of the considered combo
    target : TYPE pandas DataFrame
        DESCRIPTION. dataframe containing the target of the considered combo
    limit : TYPE int
        DESCRIPTION. integer representing the nimber of sampes to consider 
        for the training set
    even_test : TYPE boolean
        DESCRIPTION. boolean representing if the test set have to be equally distributed 
        over all the 12 months (same number of randomly extracted test samples for each month)

    Returns
    -------
    x_train_loc : TYPE pandas DataFrame
        DESCRIPTION. shuffled coherent extraction of training input data
    y_train : TYPE pandas DataFrame
        DESCRIPTION. shuffled coherent extraction of training target data
    x_test_loc : pandas DataFrame
        DESCRIPTION. shuffled coherent extraction of testing input data
    y_test : TYPE numpy array
        DESCRIPTION. shuffled coherent extraction of testing target data
    tot_idx : TYPE numpy array
        DESCRIPTION. boolean array to select the global data with the 
        same pattern as the local ones
    '''
    
    if temp_res == 'moving_monthly_avg':
        
        #adjust limit value to produce a test set that can be eavenly divided on the 12 months
        #we want to test the models month by month (same number of test samples)
        while (len(target)-limit)%12 != 0:
            #conservative choice, we increment the number of test samples until we reach a number that can be divided by 12
            limit = limit - 1
        
        idx_test = np.array(local_input.index)[limit:]
        
            
        idx_test = idx_test.astype(int)
        boolean_pattern = [True for i in range(len(target))]
        replacements = [False for i in range(len(idx_test))]
        
        for (index, replacement) in zip(idx_test, replacements):
            boolean_pattern[index] = replacement
            #reverse the training boolean pattern in order to extract the test data
            opposite_boolean_pattern = [not idx for idx in boolean_pattern]
        
        #create outputs
        x_train_loc = local_input[boolean_pattern]
        y_train = target[boolean_pattern]
        x_test_loc = local_input[opposite_boolean_pattern]
        y_test = target[opposite_boolean_pattern]
        
        return x_train_loc, y_train, x_test_loc, y_test, boolean_pattern    


##############################################################################


    

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
    except:
        pass
    
    for col in columns:
        if not(dataset[col].max() == dataset[col].min() == 0):
            dataset.loc[:,col]=(dataset.loc[:,col]-dataset.loc[:,col].mean())/dataset.loc[:,col].std()
            
    return dataset


def params_based_normalization(data,means,stds):
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
        #normalize the feature i-esima
        data[:,i] = (data[:,i] - means[i])/ stds[i]
        
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
        #compute mean and st_dev based on training data
        mean = training[:,i].mean()
        st_dev = training[:,i].std()
        #append mean and st_dev into the related lists
        means.append(mean)
        stds.append(st_dev)
        #normalization of training
        training[:,i] = (training[:,i] - mean) / st_dev
        #normalization of validation (based on training mean and st_dev)
        val_or_test[:,i] = (val_or_test[:,i] - mean) / st_dev
        
    return training, val_or_test, means, stds
    


def combo2pretty(combo):
    '''
    Parameters
    ----------
    combo : TYPE str
        DESCRIPTION. combo string composed by the path of all the variables 
        separated by '%' sign

    Returns a string of all the involved variables separated by '-'
    -------
    None.
    '''
    
    paths = combo.split('%')
    variables = ''
    for i,path in enumerate(paths):
        variable = path.split('/')[-1]
        if i != 0:
            variables = variables + '-' + variable
        elif i == 0:
            variables = variables + variable
            
    return variables



def tuple2combostring(tup):
    '''
    Parameters
    ----------
    tup : TYPE tuple
        DESCRIPTION. tuple containing the paths of all the variable of a single combination

    Returns the generating string to give as input to the generate_full_dataset function
    -------
    None.
    '''
    string = ''
    for i,item in enumerate(tup):
        if i == 0:
            string = string + item
        else:
            string = string +'%'+item
            
    return string
    


def detect_global(combo):
    
    gl_vars = ['SST','MSLP','Z500']
    
    paths = combo.split('%')
    global_paths = []
    local_paths = [] 
    for path in paths:
        if path.split('/')[-1] in gl_vars:
            global_paths.append(path)
        else:
            local_paths.append(path)
            
    local_string = ''       
    for i,path in enumerate(local_paths):
        if i == 0:
            local_string = local_string + path
        else:
            local_string = local_string + '%' + path

    return global_paths, local_string



def gen2gens(gen_string):
    '''
    Parameters
    ----------
    gen_string : TYPE str
        DESCRIPTION. string containing the paths of all the variables (global and local)

    Returns two generating strings containing the paths of the data separated by %,
    one for the local vars and the other for the global one
    -------
    None.
    '''
    
    glob_list = ['MSLP','SST','Z500']
    loc_list = ['MER','MSSHF','RH','SD','SH','t2m','TCC','TCWV','tp','UW','VW']
    
    paths = gen_string.split('%')
    glob = ''
    loc = ''
    for path in paths:
        if path.split('/')[-1] in glob_list:
            if len(glob) == 0:
                glob = glob + path
            else:
                glob = glob + '%' + path
        elif path.split('/')[-1] in loc_list:
            if len(loc) == 0:
                loc = loc + path
            else:
                loc = loc + '%' + path
                
    return loc, glob


def gen_signals(gen_string):
    '''
    Parameters
    ----------
    gen_string : TYPE str
        DESCRIPTION. string containing the paths of all the variables (global and local)

    Returns a generating scring to load the data related to climate signals (e.g NAO, ENSO, etc.)
    -------
    None.
    '''
    
    sig_list = ['NAO','SCA','EA','ENSO']
    
    paths = gen_string.split('%')
    sig = ''
    for path in paths:
        if path.split('/')[-1] in sig_list:
            if len(sig) == 0:
                sig = sig + path
            else:
                sig = sig + '%' + path
                
    return sig

            

    
    
    
    
    
    
    