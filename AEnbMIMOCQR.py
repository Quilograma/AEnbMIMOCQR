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

class AEnbMIMOCQR:

    models = [] # list of models
    residuals = [] # non-conformity set
    S_b_list = [] # list of bootstrap indexes 
    last_H_ensemble_forecasts = [] # list to store the lastest H-th forecasts
    counter = 0 # counter to know when the residuals should be updated
    qhat = [] # empirical quantile
    X_input = [] # current input to be used to make predictions
    gamma = 0 # learning rate of ACI 
    

    def __init__(self, B, alpha, phi, H, T = 0) -> None:

        # B: Number of bootstrap models
        # alpha: miscoverage rate
        # phi: aggregation function, only mean or median available
        # H: forecast horizon dimension
        # T: optional sampling of the non-conformity scores

        if not isinstance(B, int):
            raise TypeError("Value must be an integer")
        
        self.B = B

        if not isinstance(H, int):
            raise TypeError("H must be an integer")
        
        self.H = H
        
        if alpha < 0 or alpha >1:
            raise ValueError('alpha must be between 0 and 1')
        
        self.alpha = []

        for i in range(self.H):
            self.alpha.append(alpha)

        self.desired_alpha = alpha

        if phi not in ['mean','median']:

            raise ValueError("Value must be 'mean' or 'median'")
        
        self.phi = phi
        
        if not isinstance(T, int):
            raise TypeError("T must be an integer")

        self.T = T

    def fit(self, X_train, y_train, epochs):
        

        # Train b models in bootstrap datasets (bagging)
        for i in range(self.B):

            # get bootstrap indexes
            S_b = np.random.choice(X_train.shape[0], X_train.shape[0], replace=True)
            
            #Initialize the model with approriate input dim
            model = MLPQuantile(X_train.shape[1], y_train.shape[1],  quantiles=[self.alpha[0]/2, 0.5, 1-self.alpha[0]/2])

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
                    

                    yhat_list_lower.append(self.models[k].predict(X_train[i].reshape(1,-1))[0].flatten())
                    yhat_list_upper.append(self.models[k].predict(X_train[i].reshape(1,-1))[2].flatten())


                if self.phi == 'mean':
                    ensemble_forecast_lower = np.mean(np.array(yhat_list_lower), axis=0)
                    ensemble_forecast_upper = np.mean(np.array(yhat_list_upper), axis=0)
                else:
                    ensemble_forecast_lower = np.median(np.array(yhat_list_lower), axis=0)
                    ensemble_forecast_upper = np.median(np.array(yhat_list_upper), axis=0)

                non_conformity_score = np.maximum(ensemble_forecast_lower - y_train[i], y_train[i]- ensemble_forecast_upper)
                
                self.residuals.append(non_conformity_score)
        
        #compute empirical quantile

        self.qhat = np.zeros(self.H)

        for h in range(self.H):
            self.qhat[h] = np.quantile(np.array(self.residuals)[:,h], 1-self.alpha[h])    
        

        aux = list(X_train[-1]) + list(y_train[-1])
        self.X_input = aux[-X_train.shape[1]:]


        # non-conformity score sampling
        if self.T !=0 and self.T < len(self.residuals):
            indices_aux = np.random.choice(len(self.residuals), self.T, replace = False)
            self.residuals = list(np.array(self.residuals)[indices_aux])
        self.gamma = 1/len(self.residuals)


    def forecast(self):

        if self.counter == 1:
            raise Exception('Please, update with the new ground truth values before proceeding!')
        
        # list of H ensemble forecasts

        yhat_list_lower = []
        yhat_list_upper = []

        for model in self.models:
            yhat_list_lower.append(model.predict(np.array(self.X_input).reshape(1,-1))[0].flatten())
            yhat_list_upper.append(model.predict(np.array(self.X_input).reshape(1,-1))[2].flatten())

        if self.phi == 'mean':
            ensemble_forecast_lower = np.mean(np.array(yhat_list_lower), axis=0)
            ensemble_forecast_upper = np.mean(np.array(yhat_list_upper), axis=0)
        else:
            ensemble_forecast_lower = np.median(np.array(yhat_list_lower), axis=0)
            ensemble_forecast_upper = np.median(np.array(yhat_list_upper), axis=0)

        self.last_H_ensemble_forecasts = [ensemble_forecast_lower - self.qhat, ensemble_forecast_upper + self.qhat]
        self.counter+=1

        # return H-step ahead prediction intervals
        r = []
        for i in range(self.H):
            r.append([ensemble_forecast_lower[i] - self.qhat[i], ensemble_forecast_upper[i] + self.qhat[i]])


        return r
    
    #update the non-conformity score set with new scores 
    def update(self,ground_truth):
        assert len(ground_truth) == len(self.last_H_ensemble_forecasts[0])

        new_non_conformity_scores = []

        for i in range(len(ground_truth)):
            non_conformity_score = max(self.last_H_ensemble_forecasts[0][i] - ground_truth[i], ground_truth[i]- self.last_H_ensemble_forecasts[1][i])
            new_non_conformity_scores.append(non_conformity_score)

            if ground_truth[i] > self.last_H_ensemble_forecasts[0][i] and ground_truth[i] <  self.last_H_ensemble_forecasts[1][i]:
                
                self.alpha[i] = max(0,min(self.alpha[i] + self.gamma * self.desired_alpha,1))
            
            else: 
                self.alpha[i] = max(0,min(self.alpha[i] + self.gamma * (self.desired_alpha-1),1))
        

        for i in range(self.H):
            self.residuals.append(new_non_conformity_scores)
            del self.residuals[0]
            

        self.counter = 0
        self.last_H_ensemble_forecasts = []
        
        for h in range(self.H):
            self.qhat[h] = np.quantile(np.array(self.residuals)[:,h], 1-self.alpha[h]) 

        
        #update the X_input
        if len(self.X_input) > len(ground_truth):
            self.X_input = self.X_input[len(ground_truth)-len(self.X_input):] + list(ground_truth)
        else:
            self.X_input = ground_truth[-len(self.X_input):]    


