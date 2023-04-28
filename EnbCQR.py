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

class EnbCQR:

    
    models = [] # list of models
    residuals = [] # non-conformity set
    S_b_list = [] # list of bootstrap indexes 
    last_H_ensemble_forecasts = [] # list to store the lastest H-th forecasts
    counter = 0 # counter to know when the residuals should be updated
    qhat = 0 # empirical quantile
    X_input = [] # current input to be used to make predictions

    def __init__(self, B, alpha, phi, H) -> None:

        # B: Number of bootstrap models
        # alpha: miscoverage rate
        # phi: aggregation function, only mean or median available
        # H: forecast horizon dimension

        if not isinstance(B, int):
            raise TypeError("Value must be an integer")
        
        self.B = B

        if alpha < 0 or alpha >1:
            raise ValueError('alpha must be between 0 and 1')
        
        self.alpha = alpha

        if phi not in ['mean','median']:

            raise ValueError("Value must be 'mean' or 'median'")
        
        self.phi = phi

        if not isinstance(H, int):
            raise TypeError("H must be an integer")
        
        self.H = H


    def fit(self, X_train, y_train, epochs):
        

        # Train b models in bootstrap datasets (bagging)
        for i in range(self.B):

            # get bootstrap indexes
            S_b = np.random.choice(X_train.shape[0], X_train.shape[0], replace=True)
            
            #Initialize the model with approriate input dim
            model = MLPQuantile(X_train.shape[1], 1,  quantiles=[self.alpha/2, 0.5, 1-self.alpha/2])

            #fit the model on bootstrap datasets
            model.fit(X_train[S_b], y_train[S_b], epochs=epochs, verbose = 0)

            #apppend the model to the list and the bootstrap indexes
            self.models.append(model)
            self.S_b_list.append(S_b)
        
        # Compute in-sample out-of-bag non-conformity scores
        for i in range(X_train.shape[0]):
            # list to know which models incorporate the ensemble
            ensemble_list = []

            for j in range(self.B):
                #incorporate model if it did not use observation i for training
                if i not in self.S_b_list[j]:
                    ensemble_list.append(j)
            
            #if there is at least one model which did not i for training
            # produce a prediction and store the associated non-conformity score

            if len(ensemble_list)>0:
                # list of forecasts
                yhat_list_upper = []
                yhat_list_lower = []

                for k in ensemble_list:

                    yhat_list_lower.append(self.models[k].predict(X_train[i].reshape(1,-1))[0][0][0])
                    yhat_list_upper.append(self.models[k].predict(X_train[i].reshape(1,-1))[2][0][0])


                if self.phi == 'mean':
                    ensemble_forecast_lower = np.mean(yhat_list_lower)
                    ensemble_forecast_upper = np.mean(yhat_list_upper)
                else:
                    ensemble_forecast_lower = np.median(yhat_list_lower)
                    ensemble_forecast_upper = np.median(yhat_list_upper)

                non_conformity_score = max(ensemble_forecast_lower - y_train[i][0], y_train[i][0]- ensemble_forecast_upper)
                
                self.residuals.append(non_conformity_score)
        
        #compute empirical quantile
        self.qhat = np.quantile(self.residuals,1-self.alpha)
        aux = list(X_train[-1]) + list(y_train[-1])
        
        self.X_input = aux[-X_train.shape[1]:]

    def forecast(self):

        if self.counter == 1:
            raise Exception('Please, update with the new ground truth values before proceeding!')
        
        # list of H ensemble forecasts
        ensemble_PIs_list = []
        self.last_H_ensemble_forecasts = []
        
        # Deliver multi-step ahead prediction intervals
        for i in range(self.H):

            # list of forecasts
            yhat_list_lower = []
            yhat_list = []
            yhat_list_upper = []

            for model in self.models:
                yhat_list_lower.append(model.predict(np.array(self.X_input).reshape(1,-1))[0][0][0])
                yhat_list.append(model.predict(np.array(self.X_input).reshape(1,-1))[1][0][0])
                yhat_list_upper.append(model.predict(np.array(self.X_input).reshape(1,-1))[2][0][0])

            if self.phi == 'mean':
                ensemble_forecast_lower = np.mean(yhat_list_lower)
                ensemble_forecast = np.mean(yhat_list)
                ensemble_forecast_upper = np.mean(yhat_list_upper)
            else:
                ensemble_forecast_lower = np.median(yhat_list_lower)
                ensemble_forecast = np.median(yhat_list)
                ensemble_forecast_upper = np.median(yhat_list_upper)

            self.last_H_ensemble_forecasts.append([ensemble_forecast_lower, ensemble_forecast_upper])

            ensemble_PIs_list.append([ensemble_forecast_lower - self.qhat, ensemble_forecast_upper + self.qhat])
            
            self.X_input = self.X_input[1:] + [ensemble_forecast]
            self.counter+=1

        return ensemble_PIs_list
    

    #update the non-conformity score set with new scores 
    def update(self,ground_truth):
        assert len(ground_truth) == len(self.last_H_ensemble_forecasts)

        new_non_conformity_scores = []

        for i in range(len(ground_truth)):
            non_conformity_score = max(self.last_H_ensemble_forecasts[i][0] - ground_truth[i], ground_truth[i]- self.last_H_ensemble_forecasts[i][0])
            new_non_conformity_scores.append(non_conformity_score)
       

        for score in new_non_conformity_scores:
            self.residuals.append(score)
            del self.residuals[0]

        self.counter = 0
        self.qhat = np.quantile(self.residuals,1-self.alpha)

        #update the X_input
        aux = list(self.X_input) + list(ground_truth)
        self.X_input = aux[-len(self.X_input):] 


if __name__ == '__main__':

    ts = [i for i in range(100)]

    X, y = to_supervised(ts, 5, 1)

    model_enbcqr = EnbCQR(3, 0.1,'mean',2)
    
    model_enbcqr.fit(X, y, 100)
    
    for j in range(100):

        if j % model_enbcqr.H == 0 and j >0:
            print('ITERAÇÃO {}'.format(j))
            print(model_enbcqr.forecast())
            print(model_enbcqr.X_input)
            model_enbcqr.update([100+j-1, 100 + j])
            print(model_enbcqr.X_input)
            print('Updated',len(model_enbcqr.residuals), model_enbcqr.qhat)  
