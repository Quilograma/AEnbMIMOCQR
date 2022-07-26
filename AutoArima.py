import pmdarima as pm
import pandas as pd
import numpy as np

class auto_arima:
    
    def __init__(self,train,test,timesteps,alpha):
        self.train=train
        self.test=test
        self.timesteps=timesteps
        self.alpha=alpha
    
    def fit(self):

        model = pm.auto_arima(self.train,
                      m=1,             
                      d=None,           
                      seasonal=False,
                      start_p=self.timesteps,
                      max_p=self.timesteps,
                      start_q=self.timesteps,
                      max_q=self.timesteps,   
                      start_P=0,
                      method='nm',
                      alpha=self.alpha, 
                      D=None,
                      max_order =None, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
        return model
    def summary_statistics(self,arr):
    # calculates summary statistics from array
    
        return [np.quantile(arr,0.5),np.quantile(arr,0.75)-np.quantile(arr,0.25)]

    def calculate_metrics(self):
        model=self.fit()
        fc, confint = model.predict(n_periods=len(self.test), return_conf_int=True,alpha=self.alpha)

        lower_bound = confint[:, 0].flatten()
        upper_bound = confint[:, 1].flatten()

        interval_width=np.abs(upper_bound-lower_bound)
        counter=0
        coverages=[]

        for i in range(len(self.test)):
            if self.test[i] >= lower_bound[i] and self.test[i] <= upper_bound[i]:
                counter+=1
            coverages.append(counter/(i+1))
        
        return counter/len(self.test),self.summary_statistics(interval_width),coverages