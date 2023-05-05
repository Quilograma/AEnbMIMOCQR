![Image](https://github.com/Quilograma/AEnbMIMOCQR/blob/improv/fig.png?raw=true)

# What is this repository about?

This repo contains the code related to the experimental part of the paper *A General framework for multi-step ahead adaptive conformal time series forecasting*. As such, this repo contains five regression algorithms: 
- **AEnbMIMOCQR**
- **MIMOCQR**
- **EnbPI**
- **EnbCQR**
- **ARIMA** 

# Introduction
This paper presents **AEnbMIMOCQR**, a novel multi-output algorithm that aims to apply the conformal prediction procedure to generate valid multi-step ahead prediction intervals, without the need to retrain the algorithm on new data in the presence of non-exchangeable and volatile data. The proposed algorithm employs a multi-output version of CQR (Romano et al., 2019) to avoid error propagation while producing valid multi-step ahead prediction intervals. To handle distribution shifts, **AEnbMIMOCQR** uses a dynamic non-conformity set along with a homogeneous ensemble learning approach similar to EnbPI (Xu & Xie, 2021), which discards older non-conformity scores and includes the most recent ones reflecting distribution shifts. Additionally, we employ ACI (Giibs & Candes, 2021) to further enhance the adaptability of our method. Optional sampling of the non-conformity set is recommended before entering the out-of-sample phase, as decreasing the size of the non-conformity set increases the weight of the most recent non-conformity scores in empirical quantile computation.

# Design choices 
We compared all algorithms against each other in several datasets presented in the paper, where **ARIMA** is used as a baseline. All algorithms except **ARIMA** are built on top of a double hidden layered feed forward neural network, where each hidden layer contains 64 neurons. A simplied version of the proposal, **MIMOCQR**, was included to experimentally prove the benefit of employing adaptive approaches in conjunction.

# Algorithms overview

| Model | Distribution-free | Heteroscedastic | Adaptive | Multi-output |
| --- | --- | --- | --- | --- |
| **AEnbMIMOCQR** | ✔️ | ✔️ | ✔️ | ✔️ |
| **MIMOCQR** | ✔️ | ✔️ | ❌ | ✔️ |
| **EnbPI** | ✔️ | ❌ | ✔️ | ❌ |
| **EnbCQR** | ✔️ | ✔️ | ✔️ | ❌ |
| **ARIMA** | ❌ | ❌ | ❌ | ❌ |

# Requirements 
- Numpy
- Tensorflow 
- Keras
- sklearn

# How to use it

*Download the required files*

```
git clone https://github.com/Quilograma/AEnbMIMOCQR.git or download zip manually (later we may publish it on PyPI)
```

``` python

# import required libraries
from utils import to_supervised
from AEnbMIMOCQR import AEnbMIMOCQR

# univariate time series
ts = [i for i in range(1000)]

# set number of lags and forecast horizon
lags = 4
H = 5

# Convert in-sample time series to MIMO supervised structure
X, y = to_supervised(ts, n_lags = lags, n_output = H)

# Select the number of bootstrap models (B), maximum miscoverage rate (alpha), aggregation function ('mean' or 'median'), T (Number of non-conformity scores to sample in the in-sample phase)
B = 10
alpha = 0.1
phi = 'mean'
T = 100

# Initialize the algorithm
model_aenbmimocqr = AEnbMIMOCQR(B ,alpha, phi, H, T)

# fit with the desired number of epochs
epochs = 10

model_aenbmimocqr.fit(X, y, epochs = epochs)

#Obtain 5-step ahead prediction intervals each with 90% confidence.

print(model_aenbmimocqr.forecast())
# [[999.10, 1030.65], [1001.88, 1020.29], [1003.76, 1025.24], [1005.21, 1038.56], [1006.78, 1042.39]]

# update with the ground truth values at timestep t + H 

model_aenbmimocqr.update([1001,1002,1003,1004,1005]) # This will change the miscoverage rate alpha and add new-conformity scores while discarding oldest.

# We could now keep repeating this process indefinetely -> model_aenbmimocqr.forecast() -> model_aenbmimocqr.update() (...) -> model_aenbmimocqr.forecast()


```




