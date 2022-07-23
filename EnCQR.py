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

class EnCQR:

    models_list=[] # list to store B bootstrap models
    residuals_list=[] # list to store non-conformity scores
    S_b_list=[] # indexes list

    def __init__(self, B,alpha,s,X_train,y_train,X_test,y_test,timesteps,phi,epochs,batch_size):
        self.B=B
        self.alpha=alpha
        self.s=s
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.phi=phi
        self.timesteps=timesteps
        self.epochs=epochs
        self.batch_size=batch_size
    
    def Bootstrap_fit(self):
        N=self.X_train.shape[0]

        for i in range(self.B):
            S_b=np.random.choice(N,N)
            X_train_resampled,y_train_resampled=self.X_train[S_b],self.y_train[S_b]
            model=QuantileRegressionModel(X_train_resampled.shape[1],y_train_resampled.shape[1],qs=[self.alpha/2,0.5,1-self.alpha/2])
            model.fit(X_train_resampled,y_train_resampled,epochs=100,batch_size=100,verbose=0)
            self.S_b_list.append(S_b)
            self.models_list.append(model)
            
        return self.models_list

    def LOO_errors(self):

        for i in range(self.X_train.shape[0]):
            forecast_lower=[]
            forecast_upper=[]
            counter=0
            for j in range(self.B):
                if i not in self.S_b_list[j]:
                    counter+=1

                    forecast=self.models_list[j].predict(self.X_train[i].reshape(1, -1))
                    forecast_lower.append(forecast[0].flatten())
                    forecast_upper.append(forecast[2].flatten())
            actual_value=self.y_train[i]
            
            if counter>0:
                self.residuals_list.append(max(self.phi(forecast_lower)-actual_value,actual_value-self.phi(forecast_upper)))

        return self.residuals_list

    def Conf_PIs(self):
        self.Bootstrap_fit()
        self.LOO_errors()

        N=len(self.residuals_list)
        last_s_errors=[]
        conf_intervals=[]
        forecasts=[]

        for i in range(self.X_test.shape[0]):
            forecast_lower=[]
            forecast_upper=[]
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
                aux=self.models_list[j].predict(X_input)

                forecast_lower.append(aux[0].flatten())
                forecast.append(aux[1].flatten())
                forecast_upper.append(aux[2].flatten())

            ensemble_forecast_lower=self.phi(forecast_lower)
            ensemble_forecast=self.phi(forecast)
            ensemble_forecast_upper=self.phi(forecast_upper)
            forecasts.append(ensemble_forecast)

            q_yhat=np.quantile(np.array(self.residuals_list),np.floor((N+1)*(1-self.alpha))/N)

            conf_intervals.append([ensemble_forecast_lower-q_yhat, ensemble_forecast_upper+q_yhat])

            actual_value=self.y_test[i]
            error=max((ensemble_forecast_lower-q_yhat)-actual_value,actual_value-(ensemble_forecast_upper+q_yhat))
            last_s_errors.append(error)

            if (i+1)%self.s==0:

                for k in range(self.s):
                    del self.residuals_list[0]
                    self.residuals_list.append(last_s_errors[k])
                last_s_errors=[]

        return conf_intervals
