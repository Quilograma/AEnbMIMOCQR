from unittest import result
from EnbPI import EnbPI
import numpy as np
import pandas as pd
from EnCQR import EnCQR
from MIMOCQR import MIMOCQR
from AEnbMIMOCQR import AEnbMIMOCQR
from AutoArima import auto_arima
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (15,8)
plt.rcParams.update({'font.size': 30})
plt.rc('legend',fontsize=10)
sns.set_style("darkgrid", {'axes.grid' : True})

def train_val_split(X,y,N_val=400):
    N=X.shape[0]
    nval=N_val
    ntrain=N-nval

    X_train,y_train,X_val,y_val=X[:ntrain],y[:ntrain],X[-nval:],y[-nval:]

    return X_train,y_train,X_val,y_val

def phi(arr):
    return np.mean(arr)

def phimulti(arr):
    return np.mean(arr,axis=1)

def to_supervised(timeseries,n_lags,n_output=1):
    
    N=len(timeseries)
    X=np.zeros((N-n_lags-n_output+1,n_lags))
    y=np.zeros((X.shape[0],n_output))
    
    for i in range(N-n_lags):
        aux=np.zeros(n_lags)
        for j in range(i,i+n_lags,1):
            aux[j-i]=timeseries[j]
        if i+n_lags+n_output<=N:
            X[i,:]=aux
            y[i,:]=timeseries[i+n_lags:i+n_lags+n_output]

    return X,y

def summary_coverage_widths(conf_PIs,y_test_rec):
    interval_width=np.ones(len(conf_PIs))

    counter=0
    coverages=[]

    for i in range(len(conf_PIs)):
        lower_bound=conf_PIs[i][0]
        upper_bound=conf_PIs[i][1]

        if y_test_rec[i]>lower_bound and y_test_rec[i]<upper_bound:
            counter+=1
            interval_width[i]=np.abs(upper_bound-lower_bound)
        coverages.append(counter/(i+1))
    
    #interval_width=(interval_width-np.min(interval_width))/(np.max(interval_width)-np.min(interval_width))

    return counter/len(y_test_rec),summary_statistics(interval_width),coverages

def summary_statistics(arr):
    # calculates summary statistics from array
    
    return [np.median(arr),np.quantile(arr,0.75)-np.quantile(arr,0.25)]

def normalize(arr):

    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))



if __name__=='__main__':
    data=pd.read_csv('NN5.csv')
    data=data.fillna(0)
    timesteps=40
    H=30
    alpha=0.1
    B=1
    s=H
    cols=data.columns
    aux1=[]
    aux2=[]
    aux3=[]
    aux4=[]
    aux5=[]
    results_cols=['coverage','mean','std']


    for col in cols:
        print(col)
        X,y=to_supervised(normalize(data[col].values),n_lags=timesteps,n_output=H)
        X_train,y_train,X_val,y_val=train_val_split(X,y)
        #print(X_train.shape[0],X_val.shape[0])

        S_b=np.random.choice(X_train.shape[0],X_train.shape[0])
        X_train_resampled,y_train_resampled=X_train[S_b],y_train[S_b]
        n_S_B=[]
        for i in range(X_train.shape[0]):
            if i not in S_b:
                n_S_B.append(i)
        n_S_B=np.array(n_S_B)

        mimocqr=MIMOCQR(X_train_resampled,y_train_resampled,X_train[n_S_B],y_train[n_S_B],X_val,y_val,1000,100,alpha)
        results=mimocqr.calculate_coverage()
        #print(results)
        plt.plot(results[3],label='MIMOCQR')
        aux1.append([results[0],results[2][0],results[2][1]])
        aenbmimocqr=AEnbMIMOCQR(X_train,y_train,X_val,y_val,B,alpha,phi,1000,100,100)
        results=aenbmimocqr.calculate_coverage()
        plt.plot(results[3],label='AEnbMIMOCQR')
        #print(results)
        aux2.append([results[0],results[2][0],results[2][1]])

        X_rec,y_rec=to_supervised(normalize(data[col].values),n_lags=timesteps,n_output=1)
        X_train,y_train,X_val,y_val=train_val_split(X_rec,y_rec)

        enbpi=EnbPI(B,alpha,s,X_train,y_train,X_val,y_val,timesteps,phi,1000,100)
        conf_PIs=enbpi.Conf_PIs()
        results=summary_coverage_widths(conf_PIs,y_val)
        plt.plot(results[2],label='EnbPI')
        #print(results)
        aux3.append([results[0],results[1][0],results[1][1]])
        encqr=EnCQR(B,alpha,s,X_train,y_train,X_val,y_val,timesteps,phi,1000,100)
        conf_PIs=encqr.Conf_PIs()
        results=summary_coverage_widths(conf_PIs,y_val)
        plt.plot(results[2],label='EnbCQR')
        #print(results)
        aux4.append([results[0],results[1][0],results[1][1]])
        ts=normalize(data[col].values)
        arima=auto_arima(ts[:-400],ts[-400:],timesteps,alpha)
        results=arima.calculate_metrics()
        plt.plot(results[2],label='ARIMA')
        aux5.append([results[0],results[1][0],results[1][1]])
        
        plt.legend(loc='lower left')
        plt.xlabel('t')
        plt.ylabel('Coverage')
        plt.axhline(y = 1-alpha, color = 'black', linestyle = '--')
        #plt.ylim(0.5,1)
        #plt.show()
        

        lst_results=[]
        lst_results.append(np.mean(aux1,axis=0))
        lst_results.append(np.mean(aux2,axis=0))
        lst_results.append(np.mean(aux3,axis=0))
        lst_results.append(np.mean(aux4,axis=0))
        lst_results.append(np.mean(aux5,axis=0))
        df_results=pd.DataFrame(lst_results,columns=results_cols)
        print(df_results.to_latex())



 
