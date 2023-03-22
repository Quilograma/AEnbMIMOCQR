from models import MLPRegressor
import numpy as np
from utils import to_supervised

class EnbPI:

    models = []
    residuals = []
    S_b_list = []

    def __init__(self, B, alpha, phi) -> None:

        if not isinstance(B, int):
            raise TypeError("Value must be an integer")
        
        self.B = B

        if alpha < 0 or alpha >1:
            raise ValueError('alpha must be between 0 a 1')
        
        self.alpha = alpha

        if phi not in ['mean','median']:

            raise ValueError("Value must be 'mean' or 'median'")
        
        self.phi = phi

    def fit(self, X_train, y_train, epochs):
        
        # Train b models in bootstrap datasets (bagging)
        for i in range(self.B):

            S_b = np.random.choice(X_train.shape[1], X_train.shape[1], replace=True)
            
            model = MLPRegressor(X_train.shape[1], y_train.shape[1])

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


if __name__ == '__main__':

    ts = [i for i in range(100)]

    X, y = to_supervised(ts, 5, 1)

    model_enbpi = EnbPI(3, 0.1,'mean')

    model_enbpi.fit(X, y, 100)
    
    print(len(model_enbpi.residuals))                