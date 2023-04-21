import pandas as pd
from utils import to_supervised
from AEnbMIMOCQR import AEnbMIMOCQR


df = pd.read_csv("SyntheticDataset.csv")

X, y = to_supervised(df['series'].values, n_lags = 40, n_output = 30)

X_train , y_train, X_test, y_test = X[:-400], y[:-400] , X[-400:], y[-400:]

model_aenbmimocqr = AEnbMIMOCQR(10 ,0.1,'mean', 30)

model_aenbmimocqr.fit(X_train, y_train, epochs = 1000)