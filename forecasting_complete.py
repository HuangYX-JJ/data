# forecasting_complete.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ARIMA model
from statsmodels.tsa.arima.model import ARIMA

# Prophet model
from prophet import Prophet

# LSTM model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Function for ARIMA model
def arima_forecast(data):
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit.forecast(steps=5)

# Function for Prophet model
def prophet_forecast(data):
    df = pd.DataFrame(data)
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=5)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(5)

# Function for LSTM model
def lstm_forecast(data):
    X = []
    y = []
    for i in range(1, len(data)):
        X.append(data[i-1])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)
    
    return model.predict(X[-1].reshape((1, X.shape[1], 1)))

# Hybrid model
def hybrid_forecast(data):
    arima_pred = arima_forecast(data)
    prophet_pred = prophet_forecast(data)
    lstm_pred = lstm_forecast(data)
    
    # Averaging the predictions as a simple hybrid approach
    return (arima_pred + prophet_pred['yhat'].values[-1] + lstm_pred) / 3

# Example usage with synthetic data
if __name__ == "__main__":
    data = np.random.rand(100)  # Replace with actual financial metrics data
    print(hybrid_forecast(data))
