
This is the code related to the experimental part of the paper *A General framework for multi-step ahead adaptive conformal time series forecasting*. As such, this repo contains five regression algorithms: 
- **AEnbMIMOCQR**
- **MIMOCQR**
- **EnbPI**
- **EnbCQR**
- **ARIMA** 

The paper introduces **AEnbMIMOCQR**, a novel multi-output algorithm that seeks to apply the conformal prediction procedure under no exchangeable and volatile data to produce valid multi-step ahead prediction intervals without requiring retraining the algorithm on new data. **AEnbMIMOCQR** uses a multi-output version of CQR (Romano et al., 2019) to circumvent the error propagation when using the recursive strategy along with a single-output regression algorithm to produce multi-step ahead prediction intervals. Furthermore, as we want to handle distribution shifts, **AEnbMIMOCQR** uses a dynamic non-conformity set along with a homogeneous ensemble learning following the structure presented in **EnbPI** (Xu & Xie, 2021) that discards older non-conformity scores and includes the most recent that will reflect distribution shifts if that is the case. Finally, ACI (Giibs & Candes, 2021) is also employed to further improve the method's adaptability. An optional sampling of the non-conformity set while not necessary is recommended before entering the out-of-sample phase since by decreasing the size of the non-conformity set the weight of the most recent non-conformity scores will be greater on the empirical quantiles computation.
