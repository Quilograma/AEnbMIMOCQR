""" 
	author: Martim Sousa
	date:    23/03/2023
    Description: This code is an adaption of EnbCQR
    for multi-step ahead prediction intervals via
    the recursive strategy.
"""

from models import MLPQuantile
import numpy as np
from utils import to_supervised
from sklearn.model_selection import train_test_split

class MIMOCQR:

    residuals = [] # non-conformity set
    counter = 0 # counter to know when the residuals should be updated
    qhat = [] # empirical quantile
    X_input = [] # current input to be used to make predictions
    model = None
    

    def __init__(self, alpha, perc_cal, H) -> None:

        # B: Number of bootstrap models
        # alpha: miscoverage rate
        # n_cal : % of training observations used for calibration
        # phi: aggregation function, only mean or median available
        # H: forecast horizon dimension



        if not isinstance(H, int):
            raise TypeError("H must be an integer")
        
        self.H = H
        
        if alpha < 0 or alpha >1:
            raise ValueError('alpha must be between 0 and 1')
        
        self.alpha = alpha

        if perc_cal < 0 or perc_cal >1:
            raise ValueError('perc_cal must be between 0 and 1')
        
        self.perc_cal = perc_cal


        

    def fit(self, X_train, y_train, epochs):
        

        # split the training set in a new training set and a calibration set
        X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size = self.perc_cal)

            
        #Initialize the model with approriate input dim
        model = MLPQuantile(X_train.shape[1], y_train.shape[1],  quantiles=[self.alpha/2, 0.5, 1-self.alpha/2])
        
        #fit the model
        model.fit(X_train, y_train, epochs=epochs, verbose = 0)
        
        self.model = model

        
        # Compute calibration non-conformity scores
        for i in range(X_cal.shape[0]):
            
            forecast_lower = self.model.predict(X_cal[i].reshape(1,-1), verbose = 0)[0].flatten()
            forecast_upper = self.model.predict(X_cal[i].reshape(1,-1), verbose = 0)[2].flatten()
            non_conformity_score = np.maximum(forecast_lower - y_cal[i], y_cal[i]- forecast_upper)
            
            self.residuals.append(non_conformity_score)
        
        #compute empirical quantile

        self.qhat = np.zeros(self.H)

        for h in range(self.H):
            self.qhat[h] = np.quantile(np.array(self.residuals)[:,h], 1-self.alpha)    
        

        aux = list(X_train[-1]) + list(y_train[-1])
        self.X_input = aux[-X_train.shape[1]:]
        self.gamma = 1/len(self.residuals)


    def forecast(self):

        if self.counter == 1:
            raise Exception('Please, update with the new ground truth values before proceeding!')
        
        # forecast

        forecast_lower = self.model.predict(np.array(self.X_input).reshape(1,-1), verbose = 0)[0].flatten()
        forecast_upper = self.model.predict(np.array(self.X_input).reshape(1,-1), verbose = 0)[2].flatten()

        self.counter+=1

        # H-step ahead prediction intervals to return
        r = []
        for i in range(self.H):
            r.append([forecast_lower[i] - self.qhat[i], forecast_upper[i] + self.qhat[i]])
        

        return r
    
    #update the non-conformity score set with new scores 
    def update(self,ground_truth):
        assert len(ground_truth) == self.H

        self.counter = 0

        #update the X_input
        aux = list(self.X_input) + list(ground_truth)
        self.X_input = aux[-len(self.X_input):]

if __name__ == '__main__':

    ts = [i for i in range(100)]

    X, y = to_supervised(ts, 5, 2)

    model_enbcqr = MIMOCQR(0.1, 0.1, 2)
    
    model_enbcqr.fit(X, y, 100)
    
    for j in range(100):

        if j % model_enbcqr.H == 0 and j >0:
            print('ITERAÇÃO {}'.format(j))
            print(model_enbcqr.forecast())
            print(model_enbcqr.X_input)
            model_enbcqr.update([100+j-1, 100 + j])
            print(model_enbcqr.X_input)
            print('Updated',len(model_enbcqr.residuals), model_enbcqr.qhat)  
