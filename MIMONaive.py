from logging.handlers import QueueHandler
from re import I
from sklearn.utils import resample
import numpy as np
from sklearn.neural_network import MLPRegressor
#quantile loss function



class MIMONaive:

    def __init__(self,X_train,y_train,X_cal,y_cal,X_val,y_val,alpha):
        self.X_train=X_train
        self.y_train=y_train
        self.X_cal=X_cal
        self.y_cal=y_cal
        self.X_val=X_val
        self.y_val=y_val
        self.alpha=alpha
    
    def fit(self):
        model=MLPRegressor()
        model.fit(self.X_train,self.y_train)
        return model
    
    def calculate_qyhat_multi(self):
        model=self.fit()
        forecast=model.predict(self.X_cal)

        nrows=forecast.shape[0]
        ncols=forecast.shape[1]
        
        scores=np.zeros((nrows,ncols))
        q_yhats=np.zeros(ncols)
        
        for i in range(nrows):
            for j in range(ncols):
                scores[i][j]=np.abs(forecast[i][j]-self.y_cal[i][j])
        
        for j in range(ncols):
            q_yhats[j]=np.quantile(scores[:,j],np.ceil((nrows+1)*(1-self.alpha))/nrows)

        return model,q_yhats
    
    def create_conf_intervals(self):
        model,q_yhats=self.calculate_qyhat_multi()
        forecast=model.predict(self.X_val)

        nrows=forecast.shape[0]
        ncols=forecast.shape[1]
        
        lower_bounds=np.zeros((nrows,ncols))
        upper_bounds=np.zeros((nrows,ncols))
        
        for i in range(nrows):
            for j in range(ncols):
                lower_bounds[i][j]=forecast[i][j]-q_yhats[j]
                upper_bounds[i][j]=forecast[i][j]+q_yhats[j]
        
        return lower_bounds,upper_bounds


    def summary_statistics(self,arr):
        # calculates summary statistics from array
    
        return [np.quantile(arr,0.5),np.quantile(arr,0.75)-np.quantile(arr,0.25)]

    def calculate_coverage(self):
        lower_bounds,upper_bounds=self.create_conf_intervals()
        interval_sizes=np.abs(upper_bounds-lower_bounds).flatten()

        counter=0
        counter_per_horizon=np.zeros(self.y_val.shape[1])
        
        for i in range(self.y_val.shape[0]):
            for j in range(self.y_val.shape[1]):
                if lower_bounds[i][j] < self.y_val[i][j] and self.y_val[i][j] < upper_bounds[i][j]:
                    counter+=1
                    counter_per_horizon[j]+=1
        
        #interval_sizes=(interval_sizes-np.min(interval_sizes))/(np.max(interval_sizes)-np.min(interval_sizes))

        return counter/(self.y_val.shape[0]*self.y_val.shape[1]),counter_per_horizon/self.y_val.shape[0],self.summary_statistics(interval_sizes)