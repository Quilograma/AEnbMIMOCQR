""" 
	author: Martim Sousa
	date:    23/04/2023
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
from scipy.stats import norm

# read params from yaml file
with open(os.path.join(os.path.dirname(__file__), 'ExpParams.yml'), 'r') as f:
    try:
        d_params= yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)


# function to compute the IOU between two intervals
def compute_iou(interval1, interval2):
    """
    Computes the Intersection over Union (IOU) between two 1D intervals.

    Parameters:
    interval1 (tuple/list): A tuple/list representing the first interval (start, end).
    interval2 (tuple/list): A tuple/list representing the second interval (start, end).

    Returns:
    float: The IOU between the two intervals.
    """
    # Compute the intersection of the two intervals
    intersection = [max(interval1[0],interval2[0]), min(interval1[1], interval2[1])]

    if intersection[0] > intersection[1]:
        return 0
    
    # Compute the union of the two intervals
    union = [min(interval1[0],interval2[0]), max(interval1[1], interval2[1])]
    
    # Compute the IOU
    iou = abs((intersection[1] - intersection[0]) / (union[1] - union[0]))
    
    return iou


# Extract params from dictionary
B = d_params['B']
lags = d_params['lags']
H = d_params['H']
phi = d_params['phi']
T = d_params['T']
n_test = d_params['n_test']
alpha = d_params['alpha']
epochs = d_params['epochs']

# Get the multiplier coefficient 
inv_cdf = norm.ppf(1-alpha/2)

# read the dataset
df = pd.read_csv("SyntheticDataset (1).csv")

# extract train and test datasets
df_train = df[:-n_test]
df_test = df[-n_test:]


# Train ARIMA
model_arima = ARIMAModel(df_train['series'].values, alpha, H, lags)
model_arima.train()


# Convert train time series to MIMO supervised structure
X, y = to_supervised(df_train['series'].values, n_lags = lags, n_output = H)

# fit AEnbMIMOCQR and MIMOCQR
model_aenbmimocqr = AEnbMIMOCQR(B ,alpha, phi, H, T)
model_aenbmimocqr.fit(X, y, epochs = epochs)
model_mimocqr = MIMOCQR(B ,alpha, phi, H)
model_mimocqr.fit(X, y, epochs = epochs)

# Convert train time series to recursive supervised structure
X, y = to_supervised(df_train['series'].values, n_lags = lags, n_output = 1)
model_enbpi = EnbPI(B, alpha, phi, H)
model_enbcqr = EnbCQR(B, alpha, phi, H)

# fit EnbPI and EnbCQR
model_enbpi.fit(X, y, epochs = epochs)
model_enbcqr.fit(X, y, epochs = epochs)

# iou list for all models
iou_list = []



iou_list_aux = []
for i in range(0,n_test, H):
    forecast = model_arima.forecast()

    for j in range(len(forecast)):
        mu = df_test.iloc[i+j]['mean_series']
        std = df_test.iloc[i+j]['std_series']
        
        forecast_lower = forecast[j][0]
        forecast_upper = forecast[j][1]
        
        iou_list_aux.append(compute_iou([forecast_lower, forecast_upper],[mu-inv_cdf*std, mu+inv_cdf*std]))
    
    model_arima.update(df_test.iloc[i:i+30]['series'].values)



iou_list.append(iou_list_aux)


for model in [model_mimocqr, model_aenbmimocqr, model_enbpi, model_enbcqr]:
    iou_list_aux = []
    for i in range(0,n_test, H):

        forecast = model.forecast()

        for j in range(len(forecast)):
            mu = df_test.iloc[i+j]['mean_series']
            std = df_test.iloc[i+j]['std_series']

            iou_list_aux.append(compute_iou([forecast[j][0], forecast[j][1]],[mu - inv_cdf*std, mu + inv_cdf*std]))


        model.update(df_test.iloc[i:i+30]['series'].values)
    
    iou_list.append(iou_list_aux)

#average the iou
final_iou = np.mean(np.array(iou_list), axis = 1)
print(final_iou)

d = {'ARIMA': [final_iou[0]], 'MIMOCQR': [final_iou[1]], 'AEnbMIMOCQR' : [final_iou[2]], 'EnbPI': [final_iou[3]], 'EnbCQR': [final_iou[4]]}
df_final = pd.DataFrame(d)

print(df_final.to_latex())