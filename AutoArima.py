import pmdarima as pm
import numpy as np


class ARIMAModel:
    def __init__(self, time_series, alpha, horizon, p):
        self.time_series = list(time_series)
        self.alpha = alpha
        self.horizon = horizon
        self.H = horizon
        self.p = p
        self.model = None

    def train(self):
        self.model = pm.auto_arima(self.time_series,
                                   seasonal=False,
                                   p=self.p,
                                   suppress_warnings=True)
        self.model.fit(self.time_series)

    def forecast(self):
        if self.model is None:
            raise Exception('Model not trained')

        # Generate H-step ahead forecast intervals
        conf_int = self.model.predict(n_periods=self.horizon,
                                                  return_conf_int=True,
                                                  alpha=self.alpha)

        # Return confidence intervals
        return conf_int[1][-self.H:]
    
    def update(self, ground_truth):
        self.horizon = self.horizon + len(ground_truth)
    

if __name__ =='__main__':


    # Load example time series data
    data = [2,4,6,8,16,32,64,128,256,512]

    # Create ARIMA model with alpha=0.05, horizon=12, and p=2
    model = ARIMAModel(data, alpha=0.05, horizon=3, p=1)

    # Train the model
    model.train()

    # Make predictions
    conf_int = model.forecast()

    # Print corresponding intervals
    print('Confidence intervals:', conf_int)

    model.update([1024,2048])


    # Make predictions
    conf_int = model.forecast()

    print('Confidence intervals:', conf_int)

