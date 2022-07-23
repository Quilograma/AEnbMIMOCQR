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



class AEnbMIMOCQR:


    models_list=[] # list to store B bootstrap models
    residuals_list=[] # list to store non-conformity scores
    S_b_list=[]
    

    def __init__(self,X_train,y_train,X_val,y_val,B,alpha,phi,epochs,batch_size):
        self.X_train=X_train
        self.y_train=y_train
        self.X_val=X_val
        self.y_val=y_val
        self.alpha=alpha
        self.alpha_list=np.repeat(alpha,y_train.shape[1])
        self.B=B
        self.epochs=epochs
        self.batch_size=batch_size
        self.phi=phi
        
    
    



    def Bootstrap_fit(self):
        N=self.X_train.shape[0]

        for i in range(self.B):
            S_b=np.random.choice(N,N)
            X_train_resampled,y_train_resampled=self.X_train[S_b],self.y_train[S_b]
            model = QuantileRegressionModel(self.X_train.shape[1],self.y_train.shape[1],qs=[self.alpha/2, 0.5,1-self.alpha/2])
            model.fit(X_train_resampled,y_train_resampled,epochs=self.epochs,verbose=0,batch_size=self.batch_size)
            self.S_b_list.append(S_b)
            self.models_list.append(model)
            
        return self.models_list

    def LOO_errors(self):

        self.Bootstrap_fit()

        for i in range(self.X_train.shape[0]):
            forecast_lower=[]
            forecast_upper=[]
            counter=0

            for j in range(self.B):
                if i not in self.S_b_list[j]:
                    counter+=1
                    aux=self.models_list[j].predict(self.X_train[i].reshape(1, -1))
                    f_l=aux[0].flatten()
                    f_u=aux[2].flatten()
                    forecast_lower.append(f_l)
                    forecast_upper.append(f_u)

            actual_values=self.y_train[i]
            forecast_lower=np.array(forecast_lower)
            forecast_upper=np.array(forecast_upper)

            if counter >0:
                aux=[]
                for j in range(len(actual_values)):

                    aux.append(max(self.phi(forecast_lower[:,j])-actual_values[j],actual_values[j]-self.phi(forecast_upper[:,j])))
                
                self.residuals_list.append(aux)


        return self.residuals_list

    def create_conf_intervals_adaptive(self):
        self.LOO_errors()


        ncols=self.y_val.shape[1]

        q_yhats=np.zeros(ncols)

        N=len(self.residuals_list)
        # idx=np.random.choice(N, 100,replace=False)
        # self.residuals_list=np.array(self.residuals_list)[idx]
        # self.residuals_list=list(self.residuals_list)
        # N=len(self.residuals_list)
        self.gamma=(self.alpha)/N
        #N=100

        for j in range(ncols):
            q_yhats[j]=np.quantile(np.array(self.residuals_list)[:,j],(np.ceil(N+1)*(1-self.alpha_list[j]))/N)
        
        

        #adaptive part
        lower_bounds=[]
        upper_bounds=[]

        last_H_errors=[]
        last_H_cov=[]


        for i in range(self.X_val.shape[0]):

            X_input=self.X_val[i].reshape(1,-1)
            forecast_lower=[]
            forecast_upper=[]

            for k in range(len(self.models_list)):
                aux=self.models_list[k].predict(X_input)
                lb=aux[0].flatten()
                ub=aux[2].flatten()
                forecast_lower.append(lb)
                forecast_upper.append(ub)
            
            forecast_lower=np.array(forecast_lower)
            forecast_upper=np.array(forecast_upper)
            upper_bound=np.zeros(ncols)
            lower_bound=np.zeros(ncols)

            for  j in range(ncols):
                lower_bound[j]=self.phi(forecast_lower[:,j])
                upper_bound[j]=self.phi(forecast_upper[:,j])
            
            aux=[]
            aux2=[]

            for j in range(ncols):
                l=lower_bound[j]-q_yhats[j]
                u=upper_bound[j]+q_yhats[j]

                if self.y_val[i][j]<u and self.y_val[i][j]>l:
                    aux2.append(1)
                else:
                    aux2.append(0)

                aux.append(max(l-self.y_val[i][j],self.y_val[i][j]-u))
            
            last_H_errors.append(aux)
            last_H_cov.append(aux2)
            
            aux1=[]
            aux2=[]

            for j in range(ncols):
                aux1.append(lower_bound[j]-q_yhats[j])
                aux2.append(upper_bound[j]+q_yhats[j])
            
            lower_bounds.append(aux1)
            upper_bounds.append(aux2)
            
            if (i+1)%ncols==0:
                for j in range(ncols):
                    del self.residuals_list[0]
                    self.residuals_list.append(last_H_errors[j])

                    aux=np.array(last_H_cov)[:,j]
                    for i in range(len(aux)):
                        if aux[i]==1:
                            self.alpha_list[j]=self.alpha_list[j]+self.gamma*(self.alpha_list[j])
                        else:
                            self.alpha_list[j]=self.alpha_list[j]+self.gamma*(self.alpha_list[j]-1)
                    q=((np.ceil(N+1)*(1-self.alpha_list[j]))/N)
                    
                    if q>1:
                        q=1
                    elif q<0:
                        q=0 

                    q_yhats[j]=np.quantile(np.array(self.residuals_list)[:,j],q) 
               # print(self.alpha_list)            
                last_H_cov=[]
                last_H_errors=[]        
                
        return self.y_val,np.array(lower_bounds),np.array(upper_bounds)
            


    def calculate_qyhat_multi(self):
        model=self.fit()
        aux=model.predict(self.X_cal)
        forecast_lb=aux[0]
        forecast_ub=aux[2]

        nrows=forecast_lb.shape[0]

        ncols=forecast_lb.shape[1]
        
        scores=np.zeros((nrows,ncols))
        q_yhats=np.zeros(ncols)
        
        for i in range(nrows):
            for j in range(ncols):
                
                scores[i][j]=np.max([forecast_lb[i][j]-self.y_cal[i][j],self.y_cal[i][j]-forecast_ub[i][j]])
        
        for j in range(ncols):
            q_yhats[j]=np.quantile(scores[:,j],np.ceil((nrows+1)*(1-self.alpha))/nrows)

        return model,q_yhats
    
    def create_conf_intervals(self):
        model,q_yhats=self.calculate_qyhat_multi()
        aux=model.predict(self.X_val)
        forecast_lb=aux[0]
        forecast_ub=aux[2]

        nrows=forecast_lb.shape[0]
        ncols=forecast_lb.shape[1]
        
        lower_bounds=np.zeros((nrows,ncols))
        upper_bounds=np.zeros((nrows,ncols))
        
        for i in range(nrows):
            for j in range(ncols):
                lower_bounds[i][j]=forecast_lb[i][j]-q_yhats[j]
                upper_bounds[i][j]=forecast_ub[i][j]+q_yhats[j]
        
        return lower_bounds,upper_bounds 

    
    def summary_statistics(self,arr):
        # calculates summary statistics from array
    
        return [np.quantile(arr,0.5),np.quantile(arr,0.75)-np.quantile(arr,0.25)]
              
    def calculate_coverage(self):
        y_true,lower_bounds,upper_bounds= self.create_conf_intervals_adaptive()
        interval_sizes=np.abs((upper_bounds-lower_bounds)).flatten()
        counter=0
        counter_per_horizon=np.zeros(y_true.shape[1])
        
        for i in range(y_true.shape[0]):
            for j in range(y_true.shape[1]):
                if lower_bounds[i][j] < y_true[i][j] and y_true[i][j] < upper_bounds[i][j]:
                    counter+=1
                    counter_per_horizon[j]+=1
        
                    
        return counter/(y_true.shape[0]*y_true.shape[1]),counter_per_horizon/y_true.shape[0],self.summary_statistics(interval_sizes)
