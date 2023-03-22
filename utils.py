import numpy as np

# Bootstrap function
def bootstrap(data, num_samples= 0):

    if num_samples == 0:
        num_samples = data.shape[0]

    samples = np.random.choice(data.shape[0], size=num_samples, replace=True)

    return data[samples, :]

## convert a univariate time series to a supervised problem
def to_supervised(timeseries, n_lags, n_output=1):
    
    N = len(timeseries)
    X = np.zeros((N-n_lags-n_output+1, n_lags))
    y = np.zeros((X.shape[0], n_output))
    
    for i in range(N-n_lags):
        aux = np.zeros(n_lags)
        
        for j in range(i,i+n_lags, 1):
            aux[j-i] = timeseries[j]

        if i+n_lags+n_output<=N:
            X[i,:] = aux
            y[i,:] = timeseries[i+n_lags:i+n_lags+n_output]

    return X, y