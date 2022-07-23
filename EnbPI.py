from logging.handlers import QueueHandler
from sklearn.utils import resample
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

class EnbPI:

    models_list=[] # list to store B bootstrap models
    residuals_list=[] # list to store non-conformity scores
    S_b_list=[]

    def __init__(self, B,alpha,s,X_train,y_train,X_test,y_test,timesteps,phi):
        self.B=B
        self.alpha=alpha
        self.s=s
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.phi=phi
        self.timesteps=timesteps
    
    def Bootstrap_fit(self):
        N=self.X_train.shape[0]

        for i in range(self.B):
            S_b=np.random.choice(N,N)
            X_train_resampled,y_train_resampled=self.X_train[S_b],self.y_train[S_b]
            model=MLPRegressor(batch_size=100,max_iter=1000)
            model.fit(X_train_resampled,y_train_resampled)
            self.S_b_list.append(S_b)
            self.models_list.append(model)
            
        return self.models_list

    def LOO_errors(self):

        for i in range(self.X_train.shape[0]):
            forecast=[]
            counter=0
            for j in range(self.B):
                if i not in self.S_b_list[j]:
                    counter+=1

                    forecast.append(self.models_list[j].predict(self.X_train[i].reshape(1, -1))[0])
            actual_value=self.y_train[i]
            
            if counter >0:
                self.residuals_list.append(np.abs(self.phi(forecast)-actual_value))

        return self.residuals_list

    def Conf_PIs(self):
        self.Bootstrap_fit()
        self.LOO_errors()

        N=len(self.residuals_list)
        last_s_errors=[]
        conf_intervals=[]
        forecasts=[]

        for i in range(self.X_test.shape[0]):
            forecast=[]
            X_input=[]
            for k in range(self.timesteps):
                if(k+i<self.timesteps):
                    X_input.append(self.X_train[-1][k+i])
                else:
                    X_input.append(forecasts[-(self.timesteps-k-i)])
            
            X_input=np.array(X_input)
            X_input=X_input.reshape(1,-1)

            for j in range(self.B):
                forecast.append(self.models_list[j].predict(X_input)[0])

            ensemble_forecast=self.phi(forecast)
            forecasts.append(ensemble_forecast)
            q_yhat=np.quantile(self.residuals_list,np.floor((N+1)*(1-self.alpha))/N)
            conf_intervals.append([ensemble_forecast-q_yhat, ensemble_forecast+q_yhat])

            actual_value=self.y_test[i]
            error=np.abs(ensemble_forecast-actual_value)
            last_s_errors.append(error)

            if (i+1)%self.s==0:

                for k in range(self.s):
                    del self.residuals_list[0]
                    self.residuals_list.append(last_s_errors[k])
                last_s_errors=[]

        return conf_intervals
