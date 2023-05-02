""" 
	author: Martim Sousa
	date:    24/04/2023
    Description: Experiments for the synthetic dataset.
"""

# Import the required libraries
import pandas as pd
import numpy as np
import yaml
import os
from utils import to_supervised
from AEnbMIMOCQR import AEnbMIMOCQR
from MIMOCQR import MIMOCQR
from EnbPI import EnbPI
from EnbCQR import EnbCQR
from AutoArima import ARIMAModel
import json
import logging


logging.basicConfig(
    filename='my_log_file.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#Function to compute the predition interval normalized average width (PINAW)
def PINAW(PIs, y_max, y_min):
    """
    Inputs: 
    
    PIs: prediction intervals
    y_max: maximum value of time series
    y_min: minimum value of time series

    return: PINAW score
    """
    cum_width = 0
    N = len(PIs)

    for i in range(N):
        width = PIs[i][1]-PIs[i][0]
        cum_width+= width

    return cum_width/(N*(y_max-y_min))

# Function to compute the prediction interval coverage probability (PICP)
def PICP(PIs, ground_truth):
    """
    Inputs:
    PIs: prediction intervals
    
    Return: PICP score
    """
    Counter_within_PI = 0

    for i in range(len(PIs)):
        if PIs[i][0] < ground_truth[i] and ground_truth[i] < PIs[i][1]:
            Counter_within_PI+=1
    
    return Counter_within_PI/len(PIs)



# read params from yaml file
with open(os.path.join(os.path.dirname(__file__), 'ExpParams.yml'), 'r') as f:
    try:
        d_params= yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

# Extract params from dictionary
B = d_params['B']
lags = d_params['lags']
H = d_params['H']
phi = d_params['phi']
T = d_params['T']
n_test = d_params['n_test']
alpha = d_params['alpha']
epochs = d_params['epochs']



#read the dataset
df_data = pd.read_csv('NN5.csv')
df_data = df_data.fillna(0)

cols = df_data.columns

# global scores for all models
model_pinaw_global_scores = []
model_picp_global_scores = []

for col in cols:
    print(col)
    ts = df_data[col].values

    # extract train and test datasets
    ts_train = ts[:-n_test]
    ts_test = ts[-n_test:]


    # Train ARIMA
    model_arima = ARIMAModel(ts_train, alpha, H, lags)
    model_arima.train()


    # Convert train time series to MIMO supervised structure
    X, y = to_supervised(ts_train, n_lags = lags, n_output = H)

    # fit AEnbMIMOCQR and MIMOCQR
    model_aenbmimocqr = AEnbMIMOCQR(B ,alpha, phi, H, T)
    model_aenbmimocqr.fit(X, y, epochs = epochs)
    model_mimocqr = MIMOCQR(B ,alpha, phi, H)
    model_mimocqr.fit(X, y, epochs = epochs)

    # Convert train time series to recursive supervised structure
    X, y = to_supervised(ts_train, n_lags = lags, n_output = 1)
    model_enbpi = EnbPI(B, alpha, phi, H)
    model_enbcqr = EnbCQR(B, alpha, phi, H)

    # fit EnbPI and EnbCQR
    model_enbpi.fit(X, y, epochs = epochs)
    model_enbcqr.fit(X, y, epochs = epochs)

    # scores for all models
    model_pinaw_scores = []
    model_picp_scores = []

    for model in [model_arima, model_mimocqr, model_aenbmimocqr, model_enbpi, model_enbcqr]:
        PIs = []
        for i in range(0,n_test, H):

            forecast = model.forecast()

            for j in range(len(forecast)):
                PIs.append([forecast[j][0] , forecast[j][1]])

            model.update(ts_test[i:i+H])
    
        PINAW_score = PINAW(PIs, np.max(ts_train), np.min(ts_train))
        PICP_score = PICP(PIs, ts_test)

        model_pinaw_scores.append(PINAW_score)
        model_picp_scores.append(PICP_score)
    
    logging.info(f"Series {col} PINAW: {model_pinaw_scores}")
    logging.info(f"Series {col} PICP: {model_picp_scores}")

    model_pinaw_global_scores.append(model_pinaw_scores)
    model_picp_global_scores.append(model_picp_scores)

with open('results.json', 'w') as f:
    json.dump(model_pinaw_global_scores, f)
    json.dump(model_picp_global_scores, f)

model_pinaw_global_scores = np.mean(np.array(model_pinaw_global_scores), axis = 0)
model_picp_global_scores = np.mean(np.array(model_picp_global_scores), axis = 0) 

d_pinaw = {'ARIMA': [model_pinaw_global_scores[0]], 'MIMOCQR': [model_pinaw_global_scores[1]], 'AEnbMIMOCQR' : [model_pinaw_global_scores[2]], 'EnbPI': [model_pinaw_global_scores[3]], 'EnbCQR': [model_pinaw_global_scores[4]]}
df_pinaw = pd.DataFrame(d_pinaw)

print(df_pinaw.to_latex())

d_picp = {'ARIMA': [model_picp_global_scores[0]], 'MIMOCQR': [model_picp_global_scores[1]], 'AEnbMIMOCQR' : [model_picp_global_scores[2]], 'EnbPI': [model_picp_global_scores[3]], 'EnbCQR': [model_picp_global_scores[4]]}
df_picp = pd.DataFrame(d_picp)

print(df_picp.to_latex())