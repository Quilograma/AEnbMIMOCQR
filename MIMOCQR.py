from logging.handlers import QueueHandler
from sklearn.utils import resample
import numpy as np
from sklearn.neural_network import MLPRegressor
from keras import Sequential
from keras.models import Model
from keras.layers import Dense, Input,Dropout
import keras.backend as K

#quantile loss function

def tilted_loss(q,y,f):
    # q: Quantile to be evaluated, e.g., 0.5 for median.
    # y: True value.
    # f: Fitted (predicted) value.
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

# Feedforward neural network QR architecture

def QuantileRegressionModel(n_in,n_out,qs=[0.1, 0.5, 0.9]):
    ipt_layer = Input((n_in,))
    layer1 = Dense(100, activation='relu')(ipt_layer)
    drop1=Dropout(0.1)(layer1)
    layer2 = Dense(100, activation='relu')(drop1)
    drop2=Dropout(0.1)(layer2)
    
    out1 = Dense(n_out, name='out1')(drop2)
    out2 = Dense(n_out, name='out2')(drop2)
    out3 = Dense(n_out, name='out3')(drop2)
    
    q1, q2, q3 = qs
    model = Model(inputs=ipt_layer, outputs=[out1, out2, out3])
    model.compile(loss={'out1': lambda y,f: tilted_loss(q1,y,f),
                        'out2': lambda y,f: tilted_loss(q2,y,f),
                        'out3': lambda y,f: tilted_loss(q3,y,f),}, 
                  loss_weights={'out1': 1, 'out2': 1, 'out3': 1},
                 optimizer='adam')
    
    return model




class MIMOCQR:

    def __init__(self,X_train,y_train,X_cal,y_cal,X_val,y_val,epochs,batch_size,alpha):
        self.X_train=X_train
        self.y_train=y_train
        self.X_cal=X_cal
        self.y_cal=y_cal
        self.X_val=X_val
        self.y_val=y_val
        self.alpha=alpha
        self.epochs=epochs
        self.batch_size=batch_size
    
    def fit(self):
        model = QuantileRegressionModel(self.X_train.shape[1],self.y_train.shape[1],qs=[self.alpha/2, 0.5,1-self.alpha/2])
        model.fit(self.X_train,self.y_train,epochs=self.epochs,verbose=0,batch_size=self.batch_size)
        return model
    
    def calculate_qyhat_multi(self):
        model=self.fit()
        forecast=model.predict(self.X_cal)
        forecast_upper=forecast[2]
        forecast_lower=forecast[0]

        nrows=forecast_lower.shape[0]
        ncols=forecast_lower.shape[1]
        
        scores=np.zeros((nrows,ncols))
        q_yhats=np.zeros(ncols)
        
        for i in range(nrows):
            for j in range(ncols):
                scores[i][j]=max(forecast_lower[i][j]-self.y_cal[i][j],self.y_cal[i][j]-forecast_upper[i][j])
        
        for j in range(ncols):
            q_yhats[j]=np.quantile(scores[:,j],np.ceil((nrows+1)*(1-self.alpha))/nrows)

        return model,q_yhats
    
    def create_conf_intervals(self):
        model,q_yhats=self.calculate_qyhat_multi()
        forecast=model.predict(self.X_val)
        forecast_upper=forecast[2]
        forecast_lower=forecast[0]


        nrows=forecast_upper.shape[0]
        ncols=forecast_upper.shape[1]
        
        lower_bounds=np.zeros((nrows,ncols))
        upper_bounds=np.zeros((nrows,ncols))
        
        for i in range(nrows):
            for j in range(ncols):
                lower_bounds[i][j]=forecast_lower[i][j]-q_yhats[j]
                upper_bounds[i][j]=forecast_upper[i][j]+q_yhats[j]
        
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