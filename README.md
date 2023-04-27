# What is this repository about?

This repo contains the code related to the experimental part of the paper *A General framework for multi-step ahead adaptive conformal time series forecasting*. As such, this repo contains five regression algorithms: 
- **AEnbMIMOCQR**
- **MIMOCQR**
- **EnbPI**
- **EnbCQR**
- **ARIMA** 

# Introduction
This paper presents **AenbMIMOCQR**, a novel multi-output algorithm that aims to apply the conformal prediction procedure to generate valid multi-step ahead prediction intervals, without the need to retrain the algorithm on new data in the presence of non-exchangeable and volatile data. The proposed algorithm employs a multi-output version of CQR (Romano et al., 2019) to overcome error propagation when using the recursive strategy along with a single-output regression algorithm to produce multi-step ahead prediction intervals. To handle distribution shifts, AEnbMIMOCQR uses a dynamic non-conformity set along with a homogeneous ensemble learning approach similar to EnbPI (Xu & Xie, 2021), which discards older non-conformity scores and includes the most recent ones reflecting distribution shifts. Additionally, we employ ACI (Giibs & Candes, 2021) to further enhance the adaptability of our method. Optional sampling of the non-conformity set is recommended before entering the out-of-sample phase, as decreasing the size of the non-conformity set increases the weight of the most recent non-conformity scores in empirical quantile computation.

# Experimental design
We compared all algorithms against each other in several datasets presented in the paper, where **ARIMA** is used as a baseline. All algorithms except **ARIMA** are built on top of a double hidden layered feed forward neural network, where each hidden layer contains 64 neurons. A simplied version of the proposal, **MIMOCQR**, was included to experimentally prove the benefit of employing adaptive approaches in conjunction.


