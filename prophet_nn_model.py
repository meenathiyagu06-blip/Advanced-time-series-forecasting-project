"""Utilities to fit Prophet and a PyTorch residual predictor."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class ResidualMLP(nn.Module):
    def __init__(self, input_len, hidden=64, output_len=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_len)
        )

    def forward(self, x):
        # x: (batch, input_len)
        return self.net(x)

def fit_prophet(df_train, **kwargs):
    from prophet import Prophet
    m = Prophet(**kwargs)
    m.fit(df_train[['ds','y']])
    return m

def prophet_forecast(model, df, periods, freq):
    # df: historical df used as past; returns forecast df for future periods
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

def compute_residuals(df, forecast):
    # align by ds and compute residuals = y - yhat
    merged = df.merge(forecast[['ds','yhat']], on='ds', how='left')
    merged['residual'] = merged['y'] - merged['yhat']
    return merged[['ds','residual']]
