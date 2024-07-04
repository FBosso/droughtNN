# Extreme Learning Machines (ELMs) for subseasonal drought forecasting

## Brief Description:

The purpose is to build a set of monthly-based Extreme Learning Machines (ELMs) for precipitation prediction in a specific location. 

The "LOO_model_selection_combo.py" script takes into account all the combinations of the provided variables and creates, for each of these combinations, an extreme learning machine. 

Based on the paper of Giuliani et.al. (2019), since the amount of samples does not allow for a proper testing procedure, a LOO porcess is performed in order to ensure that the model is not overfitting. 
In addition to the solution proposed by Giuliani, in this case the LOO procedure is also exploited to perform both model selection and hyperparameters tuning. Indeed, the LOO procedure is repeated for each dataset among all the possible combinations of variables and for each combination of number of neurons (from 1 to 10) and activation function. At the end of the process the model with the lowest MSE is selected for each month (12 model in total) and its performances are compared with the subseasonal precipitation forecasting of the ECMWF model in that month.


## Schema

DATASET_i --> ELM + LOO  (average MSE)--> Best model of month X

