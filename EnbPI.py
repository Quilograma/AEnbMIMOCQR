from models import MLPRegressor
import numpy as np
from utils import to_supervised

class EnbPI:

    models = []
    residuals = []
    S_b_list = []
    last_H_ensemble_forecasts = []
    counter = 0
    qhat = 0
    X_input = []

    def __init__(self, B, alpha, phi, H) -> None:

        if not isinstance(B, int):
            raise TypeError("Value must be an integer")
        
        self.B = B

        if alpha < 0 or alpha >1:
            raise ValueError('alpha must be between 0 a 1')
        
        self.alpha = alpha

        if phi not in ['mean','median']:

            raise ValueError("Value must be 'mean' or 'median'")
        
        self.phi = phi

        if not isinstance(H, int):
            raise TypeError("Value must be an integer")
        
        self.H = H

    def fit(self, X_train, y_train, epochs):
        

        # Train b models in bootstrap datasets (bagging)
        for i in range(self.B):

            S_b = np.random.choice(X_train.shape[1], X_train.shape[1], replace=True)
            
            model = MLPRegressor(X_train.shape[1])

            model.fit(X_train[S_b], y_train[S_b], epochs=epochs, verbose = 0)

            self.models.append(model)
            self.S_b_list.append(S_b)


        # Compute in-sample out-of-bag non-conformity scores
        for i in range(X_train.shape[0]):
            # list to know which models incorporate the ensemble
            ensemble_list = []

            for j in range(self.B):
                if i not in self.S_b_list[j]:
                    ensemble_list.append(j)

            if len(ensemble_list)>0:
                # list of forecasts
                yhat_list = []

                for k in ensemble_list:
                    yhat_list.append(self.models[k].predict(X_train[i].reshape(1,-1))[0][0])

                if self.phi == 'mean':
                    ensemble_forecast = np.mean(yhat_list)
                else:
                    ensemble_forecast = np.median(yhat_list)

                self.residuals.append(np.abs(ensemble_forecast-y_train[i][0]))
        
        #compute empirical quantile
        self.qhat = np.quantile(self.residuals,1-self.alpha)
        self.X_input = list(X_train[-1])

    def forecast(self):

        if self.counter == 1:
            raise Exception('Please, update the non-conformity score set before proceeding!')
        
        # list of H ensemble forecasts
        ensemble_PIs_list = []

        for i in range(self.H):

            # list of forecasts
            yhat_list = []

            for model in self.models:
                yhat_list.append(model.predict(np.array(self.X_input).reshape(1,-1))[0][0])

            if self.phi == 'mean':
                ensemble_forecast = np.mean(yhat_list)
            else:
                ensemble_forecast = np.median(yhat_list)

            self.last_H_ensemble_forecasts.append(ensemble_forecast)

            ensemble_PIs_list.append([ensemble_forecast - self.qhat, ensemble_forecast + self.qhat])
            
            self.X_input = self.X_input[1:] + [ensemble_forecast]
            self.counter+=1

        return ensemble_PIs_list

    def update(self,ground_truth):
        assert len(ground_truth) == len(self.last_H_ensemble_forecasts)

        new_non_conformity_scores = np.abs(np.array(ground_truth)-np.array(self.last_H_ensemble_forecasts))

        for score in new_non_conformity_scores:
            self.residuals.append(score)
            del self.residuals[0]

        self.counter = 0
        self.last_H_ensemble_forecasts = []
        self.qhat = np.quantile(self.residuals,1-self.alpha)

        if len(self.X_input) > len(ground_truth):
            self.X_input = self.X_input[len(ground_truth)-len(self.X_input):] + ground_truth
        else:
            self.X_input = ground_truth[-len(self.X_input):]    

if __name__ == '__main__':

    ts = [i for i in range(100)]

    X, y = to_supervised(ts, 5, 1)

    model_enbpi = EnbPI(3, 0.1,'mean',2)
    
    model_enbpi.fit(X, y, 100)
    
    for j in range(100):

        if j % model_enbpi.H == 0 and j >0:
            print('ITERAÇÃO {}'.format(j))
            print(model_enbpi.forecast())
            print(model_enbpi.X_input)
            model_enbpi.update([100+j-1, 100 + j])
            print(model_enbpi.X_input)
            print ('Updated',len(model_enbpi.residuals), model_enbpi.qhat)              